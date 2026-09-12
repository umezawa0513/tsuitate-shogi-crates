use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use numpy::ndarray::{Array1, Array2, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_legality_extended::Setting;
use tsuitate_game::{Info, csa_to_piece_kind};

use crate::game_api::{ATTACK_COUNT_PIECE_KINDS, GameApi};
use crate::rl::{action_index_to_move, legal_action_indices_for_position};

#[derive(Clone, Copy, Debug)]
enum PlaneWriter {
    Board(PieceKind),
    InfoCheck,
    SelfLastCheck,
    PiecePosition,
    MoveProgress,
    Attack(PieceKind),
    AttackAtLeast(u8),
    HandAtLeast(PieceKind, u8),
    OpponentFoulAtLeast(u8),
    SelfFoulAtLeast(u8),
    LastRealTo(PieceKind),
    Capture(PieceKind),
    LastFoulFrom,
    LastFoulTo,
    LastFoulDrop,
    PlayerColor,
    LastRealFrom,
    MyCapture(PieceKind),
    LastRealToPromoted(PieceKind),
}

impl PlaneWriter {
    fn parse(value: &str) -> Result<Self, String> {
        let parts: Vec<&str> = value.split(':').collect();
        let kind = |text: &str| {
            csa_to_piece_kind(text).map_err(|_| format!("unknown piece kind {text:?}"))
        };
        let level = |text: &str| {
            text.parse::<u8>()
                .map_err(|_| format!("invalid threshold {text:?}"))
        };
        match parts.as_slice() {
            ["board", piece] => Ok(Self::Board(kind(piece)?)),
            ["info_check"] => Ok(Self::InfoCheck),
            ["self_last_check"] => Ok(Self::SelfLastCheck),
            ["piece_pos"] => Ok(Self::PiecePosition),
            ["move_progress"] => Ok(Self::MoveProgress),
            ["attack", piece] => Ok(Self::Attack(kind(piece)?)),
            ["attack_ge", threshold] => Ok(Self::AttackAtLeast(level(threshold)?)),
            ["hand_ge", piece, threshold] => Ok(Self::HandAtLeast(kind(piece)?, level(threshold)?)),
            ["opp_foul_ge", threshold] => Ok(Self::OpponentFoulAtLeast(level(threshold)?)),
            ["self_foul_ge", threshold] => Ok(Self::SelfFoulAtLeast(level(threshold)?)),
            ["last_real_to", piece] => Ok(Self::LastRealTo(kind(piece)?)),
            ["capture", piece] => Ok(Self::Capture(kind(piece)?)),
            ["last_foul_from"] => Ok(Self::LastFoulFrom),
            ["last_foul_to"] => Ok(Self::LastFoulTo),
            ["last_foul_drop"] => Ok(Self::LastFoulDrop),
            ["player_color"] => Ok(Self::PlayerColor),
            ["last_real_from"] => Ok(Self::LastRealFrom),
            ["my_capture", piece] => Ok(Self::MyCapture(kind(piece)?)),
            ["last_real_to_promoted", piece] => Ok(Self::LastRealToPromoted(kind(piece)?)),
            _ => Err(format!("unknown observation plane writer {value:?}")),
        }
    }
}

#[derive(Clone)]
struct DriverConfig {
    initial_sfen: String,
    game_kind: u8,
    promotion_rank: u8,
    initial_fouls: i8,
    draw_move_count: u16,
    foul_mask: bool,
    foul_free: bool,
    ban_first_moves: bool,
    banned_first_actions: HashSet<usize>,
    first_move_foul_loss: bool,
    board_size: usize,
    square_count: usize,
    actions_per_square: usize,
    action_count: usize,
    engine_board_stride: usize,
    engine_actions_per_square: usize,
    engine_kinds: Vec<usize>,
    engine_to_compact_kind: Vec<Option<usize>>,
    compact_move_directions: usize,
    compact_drop_kind_base: usize,
    output_to_engine_square: Vec<usize>,
    engine_to_output_square: Vec<Option<usize>>,
    writers: Vec<PlaneWriter>,
    assemble_inputs: bool,
    history_frames: usize,
    history_channels: Vec<usize>,
    real_history_frames: usize,
    real_history_channels: Vec<usize>,
    current_channels: Vec<usize>,
    input_channels: usize,
}

impl DriverConfig {
    fn color_index(color: Color) -> usize {
        if color == Color::Black { 0 } else { 1 }
    }

    fn output_square(&self, square: Square, color: Color) -> Option<usize> {
        let engine_square =
            (square.file() as usize - 1) * self.engine_board_stride + square.rank() as usize - 1;
        let absolute = *self.engine_to_output_square.get(engine_square)?.as_ref()?;
        Some(if color == Color::White {
            self.square_count - 1 - absolute
        } else {
            absolute
        })
    }

    fn engine_action_to_policy(&self, engine_action: usize, color: Color) -> Option<usize> {
        let engine_square = engine_action / self.engine_actions_per_square;
        let engine_kind = engine_action % self.engine_actions_per_square;
        let mut output_square = *self.engine_to_output_square.get(engine_square)?.as_ref()?;
        if color == Color::White {
            output_square = self.square_count - 1 - output_square;
        }
        let compact_kind = *self.engine_to_compact_kind.get(engine_kind)?.as_ref()?;
        Some(output_square * self.actions_per_square + compact_kind)
    }

    fn policy_action_to_engine(&self, policy_action: usize, color: Color) -> Option<usize> {
        if policy_action >= self.action_count {
            return None;
        }
        let mut output_square = policy_action / self.actions_per_square;
        if color == Color::White {
            output_square = self.square_count - 1 - output_square;
        }
        let compact_kind = policy_action % self.actions_per_square;
        let engine_square = *self.output_to_engine_square.get(output_square)?;
        let engine_kind = *self.engine_kinds.get(compact_kind)?;
        Some(engine_square * self.engine_actions_per_square + engine_kind)
    }

    fn same_foul_class(&self, action: usize) -> Vec<usize> {
        let square = action / self.actions_per_square;
        let kind = action % self.actions_per_square;
        if kind >= self.compact_drop_kind_base {
            return (self.compact_drop_kind_base..self.actions_per_square)
                .map(|candidate| square * self.actions_per_square + candidate)
                .collect();
        }
        let base = kind % self.compact_move_directions;
        [base, base + self.compact_move_directions]
            .into_iter()
            .filter(|candidate| *candidate < self.compact_drop_kind_base)
            .map(|candidate| square * self.actions_per_square + candidate)
            .collect()
    }
}

#[derive(Clone, Default)]
struct SideState {
    self_last_check: bool,
    last_real_move: Option<Move>,
    last_real_kind: Option<PieceKind>,
    last_real_promoted: bool,
    my_capture_kind: Option<PieceKind>,
}

#[derive(Default)]
struct GameBuffers {
    frame_bits: Vec<u16>,
    masks: Vec<u8>,
    actions: Vec<i16>,
    players: Vec<u8>,
    fouls_own: Vec<i16>,
    fouls_opp: Vec<i16>,
    plies: Vec<u16>,
    move_counts: Vec<u16>,
    infos: Vec<u8>,
    behavior_probs: Vec<f32>,
}

impl GameBuffers {
    fn steps(&self) -> usize {
        self.actions.len()
    }
}

struct PendingStep {
    color: Color,
    frame_bits: Vec<u16>,
    mask: Vec<u8>,
    fouls_own: i16,
    fouls_opp: i16,
    ply: u16,
    move_count: u16,
    first_move: bool,
}

struct GameSlot {
    game_index: usize,
    game: GameApi,
    side: [SideState; 2],
    history: [VecDeque<Vec<f32>>; 2],
    real_history: [VecDeque<Vec<f32>>; 2],
    real_last_move_count: [Option<u16>; 2],
    last_capture_kind: Option<PieceKind>,
    last_capture_board_kind: Option<PieceKind>,
    last_capture_to: Option<Square>,
    foul_excluded: Vec<bool>,
    ply: u16,
    pending: Option<PendingStep>,
    buffers: GameBuffers,
    result_black: Option<i8>,
}

struct ObservationRow {
    slot_index: usize,
    frame: Vec<f32>,
    mask: Vec<u8>,
    color: u8,
    move_count: u16,
    ply: u16,
    first_move: u8,
    input: Vec<f32>,
}

impl GameSlot {
    fn new(game_index: usize, config: &DriverConfig) -> Result<Self, String> {
        let game = GameApi::new(
            &config.initial_sfen,
            config.game_kind,
            !config.foul_free,
            config.promotion_rank,
            config.initial_fouls,
            config.initial_fouls,
            config.draw_move_count,
            None,
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            game_index,
            game,
            side: [SideState::default(), SideState::default()],
            history: std::array::from_fn(|_| VecDeque::new()),
            real_history: std::array::from_fn(|_| VecDeque::new()),
            real_last_move_count: [None, None],
            last_capture_kind: None,
            last_capture_board_kind: None,
            last_capture_to: None,
            foul_excluded: vec![false; config.action_count],
            ply: 0,
            pending: None,
            buffers: GameBuffers::default(),
            result_black: None,
        })
    }

    fn view_setting(&self, config: &DriverConfig) -> Setting {
        let mut setting = self.game.setting().clone();
        if config.foul_free {
            setting.is_tsuitate = true;
        }
        setting
    }

    fn move_count(&self) -> u16 {
        self.game.position().ply().saturating_sub(1)
    }

    fn write_square(
        frame: &mut [f32],
        channel: usize,
        square: Square,
        color: Color,
        config: &DriverConfig,
        value: f32,
    ) {
        if let Some(output_square) = config.output_square(square, color) {
            frame[channel * config.square_count + output_square] = value;
        }
    }

    fn fill_channel(frame: &mut [f32], channel: usize, config: &DriverConfig, value: f32) {
        let start = channel * config.square_count;
        frame[start..start + config.square_count].fill(value);
    }

    fn attack_value(raw: &[u8], square: Square, config: &DriverConfig) -> u8 {
        let row = square.rank() as usize - 1;
        let column = config.board_size - square.file() as usize;
        raw[row * config.board_size + column]
    }

    fn encode_frame(&self, color: Color, config: &DriverConfig) -> Vec<f32> {
        let mut frame = vec![0.0; config.writers.len() * config.square_count];
        let color_index = DriverConfig::color_index(color);
        let side = &self.side[color_index];
        let info = self.game.last_info();
        let attack_counts = self.game.attack_counts(color, true, None);
        let piece_attack_counts = self.game.attack_counts_by_piece_kind(color, true, None);
        let fouls = self.game.fouls();
        let own_remaining = if color == Color::Black {
            fouls[0]
        } else {
            fouls[1]
        };
        let opponent_remaining = if color == Color::Black {
            fouls[1]
        } else {
            fouls[0]
        };
        let self_foul_count = (config.initial_fouls as i16 - own_remaining as i16).max(0) as u8;
        let opponent_foul_count =
            (config.initial_fouls as i16 - opponent_remaining as i16).max(0) as u8;
        let last_move = self.game.last_move();
        let my_last_move =
            last_move.filter(|mv| crate::rl::infer_last_move_color(&self.game, mv) == color);

        for (channel, writer) in config.writers.iter().copied().enumerate() {
            match writer {
                PlaneWriter::Board(kind) => {
                    let mut pieces = self.game.position().piece_bitboard(Piece::new(kind, color));
                    while let Some(square) = pieces.pop() {
                        Self::write_square(&mut frame, channel, square, color, config, 1.0);
                    }
                }
                PlaneWriter::InfoCheck => {
                    if matches!(info, Some(Info::Check | Info::FoulUnderCheck)) {
                        Self::fill_channel(&mut frame, channel, config, 1.0);
                    }
                }
                PlaneWriter::SelfLastCheck => {
                    if side.self_last_check {
                        Self::fill_channel(&mut frame, channel, config, 1.0);
                    }
                }
                PlaneWriter::PiecePosition => {
                    let mut pieces = self.game.position().player_bitboard(color);
                    while let Some(square) = pieces.pop() {
                        let value = if self
                            .game
                            .position()
                            .piece_at(square)
                            .is_some_and(|piece| piece.piece_kind() == PieceKind::King)
                        {
                            2.0
                        } else {
                            1.0
                        };
                        Self::write_square(&mut frame, channel, square, color, config, value);
                    }
                }
                PlaneWriter::MoveProgress => {
                    let progress = self.move_count() as f32 / config.draw_move_count.max(1) as f32;
                    Self::fill_channel(&mut frame, channel, config, progress.clamp(0.0, 1.0));
                }
                PlaneWriter::Attack(kind) => {
                    let kind_index = ATTACK_COUNT_PIECE_KINDS
                        .iter()
                        .position(|candidate| *candidate == kind)
                        .expect("validated piece kind must have an attack plane");
                    let plane_size = config.square_count;
                    let raw = &piece_attack_counts
                        [kind_index * plane_size..(kind_index + 1) * plane_size];
                    for square in Square::all() {
                        if config.output_square(square, color).is_some()
                            && Self::attack_value(raw, square, config) > 0
                        {
                            Self::write_square(&mut frame, channel, square, color, config, 1.0);
                        }
                    }
                }
                PlaneWriter::AttackAtLeast(threshold) => {
                    for square in Square::all() {
                        if config.output_square(square, color).is_some()
                            && Self::attack_value(&attack_counts, square, config) >= threshold
                        {
                            Self::write_square(&mut frame, channel, square, color, config, 1.0);
                        }
                    }
                }
                PlaneWriter::HandAtLeast(kind, threshold) => {
                    let held = self
                        .game
                        .position()
                        .hand(Piece::new(kind, color))
                        .unwrap_or(0);
                    if held >= threshold {
                        Self::fill_channel(&mut frame, channel, config, 1.0);
                    }
                }
                PlaneWriter::OpponentFoulAtLeast(threshold) => {
                    if opponent_foul_count >= threshold {
                        Self::fill_channel(&mut frame, channel, config, 1.0);
                    }
                }
                PlaneWriter::SelfFoulAtLeast(threshold) => {
                    if self_foul_count >= threshold {
                        Self::fill_channel(&mut frame, channel, config, 1.0);
                    }
                }
                PlaneWriter::LastRealTo(kind) => {
                    if side.last_real_kind == Some(kind)
                        && let Some(mv) = side.last_real_move
                    {
                        Self::write_square(&mut frame, channel, move_to(mv), color, config, 1.0);
                    }
                }
                PlaneWriter::Capture(kind) => {
                    if self.last_capture_kind == Some(kind)
                        && let Some(square) = self.last_capture_to
                    {
                        Self::write_square(&mut frame, channel, square, color, config, 1.0);
                    }
                }
                PlaneWriter::LastFoulFrom => {
                    if let Some(Move::Normal { from, .. }) = my_last_move {
                        Self::write_square(&mut frame, channel, from, color, config, 1.0);
                    }
                }
                PlaneWriter::LastFoulTo => {
                    if let Some(Move::Normal { to, .. }) = my_last_move {
                        Self::write_square(&mut frame, channel, to, color, config, 1.0);
                    }
                }
                PlaneWriter::LastFoulDrop => {
                    if let Some(Move::Drop { to, .. }) = my_last_move {
                        Self::write_square(&mut frame, channel, to, color, config, 1.0);
                    }
                }
                PlaneWriter::PlayerColor => {
                    if color == Color::Black {
                        Self::fill_channel(&mut frame, channel, config, 1.0);
                    }
                }
                PlaneWriter::LastRealFrom => {
                    if let Some(Move::Normal { from, .. }) = side.last_real_move {
                        Self::write_square(&mut frame, channel, from, color, config, 1.0);
                    }
                }
                PlaneWriter::MyCapture(kind) => {
                    if side.my_capture_kind == Some(kind)
                        && let Some(mv) = side.last_real_move
                    {
                        Self::write_square(&mut frame, channel, move_to(mv), color, config, 1.0);
                    }
                }
                PlaneWriter::LastRealToPromoted(kind) => {
                    if side.last_real_kind == Some(kind)
                        && let Some(mv) = side.last_real_move
                    {
                        let value = if side.last_real_promoted { 2.0 } else { 1.0 };
                        Self::write_square(&mut frame, channel, move_to(mv), color, config, value);
                    }
                }
            }
        }
        frame
    }

    fn legal_mask(&self, color: Color, config: &DriverConfig) -> Vec<u8> {
        let setting = self.view_setting(config);
        let legal = legal_action_indices_for_position(self.game.position(), &setting, None);
        let mut mask = vec![0; config.action_count];
        for engine_action in legal {
            if let Some(action) = config.engine_action_to_policy(engine_action, color) {
                mask[action] = 1;
            }
        }
        if config.ban_first_moves
            && self.side[DriverConfig::color_index(color)]
                .last_real_move
                .is_none()
        {
            for action in &config.banned_first_actions {
                mask[*action] = 0;
            }
        }
        if self.foul_excluded.iter().any(|excluded| *excluded) {
            let mut pruned = mask.clone();
            for (allowed, excluded) in pruned.iter_mut().zip(&self.foul_excluded) {
                if *excluded {
                    *allowed = 0;
                }
            }
            let any = pruned.iter().any(|allowed| *allowed != 0);
            if (config.foul_mask && (any || config.foul_free))
                || (!config.foul_mask && config.foul_free && !any)
            {
                mask = pruned;
            }
        }
        mask
    }

    fn append_selected(
        output: &mut Vec<f32>,
        frame: &[f32],
        channels: &[usize],
        square_count: usize,
    ) {
        for channel in channels {
            let start = channel * square_count;
            output.extend_from_slice(&frame[start..start + square_count]);
        }
    }

    fn assemble_input(
        &mut self,
        color: Color,
        move_count: u16,
        frame: &[f32],
        config: &DriverConfig,
    ) -> Vec<f32> {
        if !config.assemble_inputs {
            return Vec::new();
        }
        let color_index = DriverConfig::color_index(color);
        let history = &mut self.history[color_index];
        history.push_back(frame.to_vec());
        while history.len() > config.history_frames + 1 {
            history.pop_front();
        }
        let real_history = &mut self.real_history[color_index];
        if config.real_history_frames > 0
            && self.real_last_move_count[color_index] != Some(move_count)
        {
            real_history.push_back(frame.to_vec());
            while real_history.len() > config.real_history_frames + 1 {
                real_history.pop_front();
            }
            self.real_last_move_count[color_index] = Some(move_count);
        }

        let mut input = Vec::with_capacity(config.input_channels * config.square_count);
        let available_real = real_history.len().saturating_sub(1);
        let real_take = available_real.min(config.real_history_frames);
        input.resize(
            (config.real_history_frames - real_take)
                * config.real_history_channels.len()
                * config.square_count,
            0.0,
        );
        for historical in real_history
            .iter()
            .take(available_real)
            .skip(available_real - real_take)
        {
            Self::append_selected(
                &mut input,
                historical,
                &config.real_history_channels,
                config.square_count,
            );
        }

        let available = history.len().saturating_sub(1);
        let take = available.min(config.history_frames);
        input.resize(
            input.len()
                + (config.history_frames - take)
                    * config.history_channels.len()
                    * config.square_count,
            0.0,
        );
        for historical in history.iter().take(available).skip(available - take) {
            Self::append_selected(
                &mut input,
                historical,
                &config.history_channels,
                config.square_count,
            );
        }
        Self::append_selected(
            &mut input,
            frame,
            &config.current_channels,
            config.square_count,
        );
        debug_assert_eq!(input.len(), config.input_channels * config.square_count);
        input
    }

    fn observe(
        &mut self,
        slot_index: usize,
        config: &DriverConfig,
    ) -> Result<Option<ObservationRow>, String> {
        if self.pending.is_some() {
            return Err("observe_batch called twice without apply_batch".to_string());
        }
        let color = self.game.position().side_to_move();
        let frame = self.encode_frame(color, config);
        let mask = self.legal_mask(color, config);
        if !mask.iter().any(|allowed| *allowed != 0) {
            self.result_black = Some(if color == Color::Black { -1 } else { 1 });
            return Ok(None);
        }
        let fouls = self.game.fouls();
        let (fouls_own, fouls_opp) = if color == Color::Black {
            (fouls[0], fouls[1])
        } else {
            (fouls[1], fouls[0])
        };
        let first_move = self.side[DriverConfig::color_index(color)]
            .last_real_move
            .is_none();
        let move_count = self.move_count();
        let input = self.assemble_input(color, move_count, &frame, config);
        let frame_bits = frame.iter().map(|value| f32_to_f16_bits(*value)).collect();
        self.pending = Some(PendingStep {
            color,
            frame_bits,
            mask: mask.clone(),
            fouls_own: fouls_own as i16,
            fouls_opp: fouls_opp as i16,
            ply: self.ply,
            move_count,
            first_move,
        });
        Ok(Some(ObservationRow {
            slot_index,
            frame,
            mask,
            color: DriverConfig::color_index(color) as u8,
            move_count,
            ply: self.ply,
            first_move: u8::from(first_move),
            input,
        }))
    }

    fn apply(
        &mut self,
        action: usize,
        behavior_prob: f32,
        config: &DriverConfig,
    ) -> Result<(i8, u8, u8), String> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| "apply_batch called before observe_batch".to_string())?;
        if pending.mask.get(action).copied() != Some(1) {
            return Err(format!(
                "action {action} is not present in the current legal mask"
            ));
        }
        let engine_action = config
            .policy_action_to_engine(action, pending.color)
            .ok_or_else(|| format!("policy action {action} cannot be mapped to the engine"))?;
        let setting = self.view_setting(config);
        let mv = action_index_to_move(self.game.position(), setting.is_tsuitate, engine_action)
            .ok_or_else(|| format!("engine action {engine_action} cannot be decoded"))?;
        let captured_board_kind = match mv {
            Move::Normal { to, .. } => self
                .game
                .position()
                .piece_at(to)
                .map(|piece| piece.piece_kind()),
            Move::Drop { .. } => None,
        };
        let side_index = DriverConfig::color_index(pending.color);
        let is_banned_first_move = config.first_move_foul_loss
            && pending.first_move
            && config.banned_first_actions.contains(&action);
        let moved = self.game.make_move_raw(mv);
        self.ply = self.ply.saturating_add(1);

        let (info, result_black) = if is_banned_first_move {
            (
                Info::LossByFoul,
                Some(if pending.color == Color::Black { -1 } else { 1 }),
            )
        } else if !moved {
            if !config.foul_free {
                return Err(format!("engine rejected visible action {engine_action}"));
            }
            for equivalent in config.same_foul_class(action) {
                self.foul_excluded[equivalent] = true;
            }
            (Info::None, None)
        } else {
            let info = self
                .game
                .last_info()
                .ok_or_else(|| "engine accepted an action without last_info".to_string())?;
            self.side[side_index].self_last_check = info == Info::Check;
            if matches!(info, Info::Foul | Info::FoulUnderCheck) {
                if config.foul_mask {
                    for equivalent in config.same_foul_class(action) {
                        self.foul_excluded[equivalent] = true;
                    }
                }
            } else {
                self.foul_excluded.fill(false);
                let kind = self
                    .game
                    .position()
                    .piece_at(move_to(mv))
                    .map(|piece| piece.piece_kind());
                let compact_kind = action % config.actions_per_square;
                let promoted = compact_kind >= config.compact_move_directions
                    && compact_kind < config.compact_drop_kind_base;
                let capture_kind = self.game.last_capture_piece_kind();
                let side = &mut self.side[side_index];
                side.last_real_move = Some(mv);
                side.last_real_kind = kind;
                side.last_real_promoted = promoted;
                side.my_capture_kind = capture_kind;
                if let Some(capture_kind) = capture_kind {
                    self.last_capture_kind = Some(capture_kind);
                    self.last_capture_board_kind = captured_board_kind.or(Some(capture_kind));
                    self.last_capture_to = Some(move_to(mv));
                } else {
                    self.last_capture_kind = None;
                    self.last_capture_board_kind = None;
                    self.last_capture_to = None;
                }
            }
            let result = match info {
                Info::Checkmate => Some(if pending.color == Color::Black { 1 } else { -1 }),
                Info::LossByFoul => Some(if pending.color == Color::Black { -1 } else { 1 }),
                Info::Draw => Some(0),
                _ => None,
            };
            (info, result)
        };

        self.buffers.frame_bits.extend(pending.frame_bits);
        self.buffers.masks.extend(pending.mask);
        self.buffers.actions.push(action as i16);
        self.buffers.players.push(side_index as u8);
        self.buffers.fouls_own.push(pending.fouls_own);
        self.buffers.fouls_opp.push(pending.fouls_opp);
        self.buffers.plies.push(pending.ply);
        self.buffers.move_counts.push(pending.move_count);
        self.buffers.infos.push(info as u8);
        self.buffers.behavior_probs.push(behavior_prob);
        let done = u8::from(result_black.is_some());
        let reward = match result_black {
            Some(0) | None => 0,
            Some(result)
                if (pending.color == Color::Black && result > 0)
                    || (pending.color == Color::White && result < 0) =>
            {
                1
            }
            Some(_) => -1,
        };
        if let Some(result) = result_black {
            self.result_black = Some(result);
        }
        Ok((reward, done, info as u8))
    }
}

fn move_to(mv: Move) -> Square {
    match mv {
        Move::Normal { to, .. } | Move::Drop { to, .. } => to,
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

struct FinishedGame {
    game_index: usize,
    buffers: GameBuffers,
    result_black: i8,
}

#[pyclass(name = "SelfPlayBatch")]
pub(crate) struct PySelfPlayBatch {
    config: DriverConfig,
    game_count: usize,
    batch_size: usize,
    next_game: usize,
    slots: Vec<Option<GameSlot>>,
    finished: Vec<Option<FinishedGame>>,
    pending_slots: Vec<usize>,
    pool: Option<rayon::ThreadPool>,
    taken: bool,
    observe_seconds: f64,
    apply_seconds: f64,
    pack_seconds: f64,
}

#[pymethods]
impl PySelfPlayBatch {
    #[new]
    #[pyo3(signature = (
        game_count,
        batch_size,
        initial_sfen,
        game_kind,
        promotion_rank,
        initial_fouls,
        draw_move_count,
        foul_mask,
        foul_free,
        ban_first_moves,
        banned_first_actions,
        first_move_foul_loss,
        board_size,
        actions_per_square,
        action_count,
        engine_board_stride,
        engine_actions_per_square,
        engine_kinds,
        compact_move_directions,
        compact_drop_kind_base,
        output_to_engine_square,
        plane_writers,
        assemble_inputs,
        history_frames,
        history_channels,
        real_history_frames,
        real_history_channels,
        current_channels,
        thread_count=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        game_count: usize,
        batch_size: usize,
        initial_sfen: String,
        game_kind: u8,
        promotion_rank: u8,
        initial_fouls: i8,
        draw_move_count: u16,
        foul_mask: bool,
        foul_free: bool,
        ban_first_moves: bool,
        banned_first_actions: Vec<usize>,
        first_move_foul_loss: bool,
        board_size: usize,
        actions_per_square: usize,
        action_count: usize,
        engine_board_stride: usize,
        engine_actions_per_square: usize,
        engine_kinds: Vec<usize>,
        compact_move_directions: usize,
        compact_drop_kind_base: usize,
        output_to_engine_square: Vec<usize>,
        plane_writers: Vec<(String, usize)>,
        assemble_inputs: bool,
        history_frames: usize,
        history_channels: Vec<usize>,
        real_history_frames: usize,
        real_history_channels: Vec<usize>,
        current_channels: Vec<usize>,
        thread_count: Option<usize>,
    ) -> PyResult<Self> {
        if game_count == 0 {
            return Err(PyValueError::new_err("game_count must be positive"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        if board_size == 0 || actions_per_square == 0 || engine_actions_per_square == 0 {
            return Err(PyValueError::new_err(
                "board and action dimensions must be positive",
            ));
        }
        let square_count = board_size
            .checked_mul(board_size)
            .ok_or_else(|| PyValueError::new_err("board size overflow"))?;
        if action_count != square_count * actions_per_square {
            return Err(PyValueError::new_err(
                "action_count does not match board dimensions",
            ));
        }
        if engine_kinds.len() != actions_per_square {
            return Err(PyValueError::new_err(
                "engine_kinds must have actions_per_square entries",
            ));
        }
        if output_to_engine_square.len() != square_count {
            return Err(PyValueError::new_err(
                "output_to_engine_square has the wrong length",
            ));
        }
        let engine_square_count = engine_board_stride
            .checked_mul(engine_board_stride)
            .ok_or_else(|| PyValueError::new_err("engine board stride overflow"))?;
        let mut engine_to_output_square = vec![None; engine_square_count];
        for (output, engine) in output_to_engine_square.iter().copied().enumerate() {
            if engine >= engine_square_count || engine_to_output_square[engine].is_some() {
                return Err(PyValueError::new_err("output/engine square map is invalid"));
            }
            engine_to_output_square[engine] = Some(output);
        }
        let mut engine_to_compact_kind = vec![None; engine_actions_per_square];
        for (compact, engine) in engine_kinds.iter().copied().enumerate() {
            if engine >= engine_actions_per_square || engine_to_compact_kind[engine].is_some() {
                return Err(PyValueError::new_err("engine kind map is invalid"));
            }
            engine_to_compact_kind[engine] = Some(compact);
        }
        let mut ordered = plane_writers;
        ordered.sort_by_key(|(_, index)| *index);
        if ordered
            .iter()
            .enumerate()
            .any(|(expected, (_, actual))| expected != *actual)
        {
            return Err(PyValueError::new_err(
                "plane writer indices must be contiguous",
            ));
        }
        let writers = ordered
            .iter()
            .map(|(name, _)| PlaneWriter::parse(name))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        if history_channels
            .iter()
            .chain(real_history_channels.iter())
            .chain(current_channels.iter())
            .any(|channel| *channel >= writers.len())
        {
            return Err(PyValueError::new_err(
                "input assembly channel is out of range",
            ));
        }
        let input_channels = history_frames
            .checked_mul(history_channels.len())
            .and_then(|value| {
                real_history_frames
                    .checked_mul(real_history_channels.len())
                    .and_then(|real| value.checked_add(real))
            })
            .and_then(|value| value.checked_add(current_channels.len()))
            .ok_or_else(|| PyValueError::new_err("input channel count overflow"))?;
        if assemble_inputs && input_channels == 0 {
            return Err(PyValueError::new_err(
                "assembled input must contain at least one channel",
            ));
        }
        let banned_first_actions = banned_first_actions.into_iter().collect::<HashSet<_>>();
        if banned_first_actions
            .iter()
            .any(|action| *action >= action_count)
        {
            return Err(PyValueError::new_err("banned first action is out of range"));
        }
        if first_move_foul_loss && foul_free {
            return Err(PyValueError::new_err(
                "first_move_foul_loss and foul_free cannot be enabled together",
            ));
        }
        let config = DriverConfig {
            initial_sfen,
            game_kind,
            promotion_rank,
            initial_fouls,
            draw_move_count,
            foul_mask,
            foul_free,
            ban_first_moves,
            banned_first_actions,
            first_move_foul_loss,
            board_size,
            square_count,
            actions_per_square,
            action_count,
            engine_board_stride,
            engine_actions_per_square,
            engine_kinds,
            engine_to_compact_kind,
            compact_move_directions,
            compact_drop_kind_base,
            output_to_engine_square,
            engine_to_output_square,
            writers,
            assemble_inputs,
            history_frames,
            history_channels,
            real_history_frames,
            real_history_channels,
            current_channels,
            input_channels,
        };
        let pool = thread_count
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .transpose()?;
        let active_count = game_count.min(batch_size);
        let mut slots = Vec::with_capacity(batch_size);
        for game_index in 0..active_count {
            slots.push(Some(
                GameSlot::new(game_index, &config).map_err(PyValueError::new_err)?,
            ));
        }
        slots.resize_with(batch_size, || None);
        let finished = (0..game_count).map(|_| None).collect();
        Ok(Self {
            config,
            game_count,
            batch_size,
            next_game: active_count,
            slots,
            finished,
            pending_slots: Vec::new(),
            pool,
            taken: false,
            observe_seconds: 0.0,
            apply_seconds: 0.0,
            pack_seconds: 0.0,
        })
    }

    #[getter]
    fn done(&self) -> bool {
        self.finished.iter().all(Option::is_some)
    }

    #[getter]
    fn completed_games(&self) -> usize {
        self.finished.iter().filter(|game| game.is_some()).count()
    }

    #[getter]
    fn frame_channels(&self) -> usize {
        self.config.writers.len()
    }

    fn observe_batch<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        let started = Instant::now();
        if !self.pending_slots.is_empty() {
            return Err(PyRuntimeError::new_err(
                "observe_batch called before applying the previous batch",
            ));
        }
        let results = py.detach(|| {
            let config = &self.config;
            let slots = &mut self.slots;
            let mut build = || {
                slots
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(slot_index, slot)| {
                        slot.as_mut().map(|slot| slot.observe(slot_index, config))
                    })
                    .collect::<Vec<_>>()
            };
            if let Some(pool) = &self.pool {
                pool.install(build)
            } else {
                build()
            }
        });
        let mut rows = Vec::new();
        for result in results {
            if let Some(row) = result.map_err(PyRuntimeError::new_err)? {
                rows.push(row);
            }
        }
        self.finish_completed_slots()?;
        rows.sort_by_key(|row| row.slot_index);
        self.pending_slots = rows.iter().map(|row| row.slot_index).collect();
        let row_count = rows.len();
        let mut frames =
            Vec::with_capacity(row_count * self.config.writers.len() * self.config.square_count);
        let mut inputs =
            Vec::with_capacity(row_count * self.config.input_channels * self.config.square_count);
        let mut masks = Vec::with_capacity(row_count * self.config.action_count);
        let mut slot_ids = Vec::with_capacity(row_count);
        let mut colors = Vec::with_capacity(row_count);
        let mut move_counts = Vec::with_capacity(row_count);
        let mut plies = Vec::with_capacity(row_count);
        let mut first_moves = Vec::with_capacity(row_count);
        for row in rows {
            frames.extend(row.frame);
            inputs.extend(row.input);
            masks.extend(row.mask);
            slot_ids.push(row.slot_index as i64);
            colors.push(row.color as i64);
            move_counts.push(row.move_count as i64);
            plies.push(row.ply as i64);
            first_moves.push(row.first_move);
        }
        let result = (
            Array4::from_shape_vec(
                (
                    row_count,
                    self.config.writers.len(),
                    self.config.board_size,
                    self.config.board_size,
                ),
                frames,
            )
            .expect("observation frame shape is internally consistent")
            .into_pyarray(py),
            Array4::from_shape_vec(
                (
                    row_count,
                    if self.config.assemble_inputs {
                        self.config.input_channels
                    } else {
                        0
                    },
                    self.config.board_size,
                    self.config.board_size,
                ),
                inputs,
            )
            .expect("assembled input shape is internally consistent")
            .into_pyarray(py),
            Array2::from_shape_vec((row_count, self.config.action_count), masks)
                .expect("legal mask shape is internally consistent")
                .into_pyarray(py),
            Array1::from_vec(slot_ids).into_pyarray(py),
            Array1::from_vec(colors).into_pyarray(py),
            Array1::from_vec(move_counts).into_pyarray(py),
            Array1::from_vec(plies).into_pyarray(py),
            Array1::from_vec(first_moves).into_pyarray(py),
        );
        self.observe_seconds += started.elapsed().as_secs_f64();
        Ok(result)
    }

    fn apply_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
        behavior_probs: Vec<f32>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        let started = Instant::now();
        if actions.len() != self.pending_slots.len() || behavior_probs.len() != actions.len() {
            return Err(PyValueError::new_err(
                "actions and behavior_probs must match the observed batch length",
            ));
        }
        let mut action_by_slot = vec![None; self.batch_size];
        for (row, slot_index) in self.pending_slots.iter().copied().enumerate() {
            action_by_slot[slot_index] = Some((actions[row], behavior_probs[row]));
        }
        let config = &self.config;
        let slots = &mut self.slots;
        let mut apply = || {
            slots
                .par_iter_mut()
                .enumerate()
                .filter_map(|(slot_index, slot)| {
                    let (action, probability) = action_by_slot[slot_index]?;
                    Some(match slot.as_mut() {
                        Some(slot) => slot.apply(action, probability, config),
                        None => Err("observed slot disappeared before apply".to_string()),
                    })
                })
                .collect::<Vec<_>>()
        };
        let results = py.detach(|| {
            if let Some(pool) = &self.pool {
                pool.install(apply)
            } else {
                apply()
            }
        });
        let mut rewards = Vec::with_capacity(results.len());
        let mut dones = Vec::with_capacity(results.len());
        let mut infos = Vec::with_capacity(results.len());
        for result in results {
            let (reward, done, info) = result.map_err(PyRuntimeError::new_err)?;
            rewards.push(reward);
            dones.push(done);
            infos.push(info);
        }
        self.pending_slots.clear();
        self.finish_completed_slots()?;
        let result = (
            Array1::from_vec(rewards).into_pyarray(py),
            Array1::from_vec(dones).into_pyarray(py),
            Array1::from_vec(infos).into_pyarray(py),
        );
        self.apply_seconds += started.elapsed().as_secs_f64();
        Ok(result)
    }

    fn take_records<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let started = Instant::now();
        if !self.done() {
            return Err(PyRuntimeError::new_err(
                "take_records requires every requested game to be complete",
            ));
        }
        if self.taken {
            return Err(PyRuntimeError::new_err(
                "take_records may only be called once",
            ));
        }
        self.taken = true;
        let mut frames = Vec::new();
        let mut masks = Vec::new();
        let mut actions = Vec::new();
        let mut players = Vec::new();
        let mut fouls_own = Vec::new();
        let mut fouls_opp = Vec::new();
        let mut plies = Vec::new();
        let mut move_counts = Vec::new();
        let mut infos = Vec::new();
        let mut behavior_probs = Vec::new();
        let mut rewards = Vec::new();
        let mut offsets = Vec::with_capacity(self.game_count + 1);
        let mut results = Vec::with_capacity(self.game_count);
        offsets.push(0);
        for (expected, game) in self.finished.iter_mut().enumerate() {
            let finished = game
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("missing finished game"))?;
            if finished.game_index != expected {
                return Err(PyRuntimeError::new_err("finished game order is corrupted"));
            }
            let steps = finished.buffers.steps();
            frames.extend(finished.buffers.frame_bits);
            masks.extend(finished.buffers.masks);
            actions.extend(finished.buffers.actions);
            players.extend(finished.buffers.players);
            fouls_own.extend(finished.buffers.fouls_own);
            fouls_opp.extend(finished.buffers.fouls_opp);
            plies.extend(finished.buffers.plies);
            move_counts.extend(finished.buffers.move_counts);
            infos.extend(finished.buffers.infos);
            behavior_probs.extend(finished.buffers.behavior_probs);
            let previous = rewards.len();
            rewards.resize(previous + steps, 0.0);
            if steps > 0 {
                rewards[previous + steps - 1] = finished.result_black as f32;
            }
            offsets.push((previous + steps) as i64);
            results.push(finished.result_black);
        }
        let total_steps = actions.len();
        let packed = PyDict::new(py);
        packed.set_item(
            "frame_bits",
            Array4::from_shape_vec(
                (
                    total_steps,
                    self.config.writers.len(),
                    self.config.board_size,
                    self.config.board_size,
                ),
                frames,
            )
            .expect("packed frame shape is internally consistent")
            .into_pyarray(py),
        )?;
        packed.set_item(
            "masks",
            Array2::from_shape_vec((total_steps, self.config.action_count), masks)
                .expect("packed mask shape is internally consistent")
                .into_pyarray(py),
        )?;
        packed.set_item("actions", Array1::from_vec(actions).into_pyarray(py))?;
        packed.set_item("players", Array1::from_vec(players).into_pyarray(py))?;
        packed.set_item("fouls_own", Array1::from_vec(fouls_own).into_pyarray(py))?;
        packed.set_item("fouls_opp", Array1::from_vec(fouls_opp).into_pyarray(py))?;
        packed.set_item("plies", Array1::from_vec(plies).into_pyarray(py))?;
        packed.set_item(
            "move_counts",
            Array1::from_vec(move_counts).into_pyarray(py),
        )?;
        packed.set_item("infos", Array1::from_vec(infos).into_pyarray(py))?;
        packed.set_item(
            "behavior_probs",
            Array1::from_vec(behavior_probs).into_pyarray(py),
        )?;
        packed.set_item("rewards", Array1::from_vec(rewards).into_pyarray(py))?;
        packed.set_item("offsets", Array1::from_vec(offsets).into_pyarray(py))?;
        packed.set_item("result_black", Array1::from_vec(results).into_pyarray(py))?;
        self.pack_seconds += started.elapsed().as_secs_f64();
        Ok(packed)
    }

    fn timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let timings = PyDict::new(py);
        timings.set_item("observe_seconds", self.observe_seconds)?;
        timings.set_item("apply_seconds", self.apply_seconds)?;
        timings.set_item("pack_seconds", self.pack_seconds)?;
        Ok(timings)
    }
}

impl PySelfPlayBatch {
    fn finish_completed_slots(&mut self) -> PyResult<()> {
        for slot_index in 0..self.slots.len() {
            let should_finish = self.slots[slot_index]
                .as_ref()
                .is_some_and(|slot| slot.result_black.is_some());
            if !should_finish {
                continue;
            }
            let slot = self.slots[slot_index]
                .take()
                .expect("checked occupied slot");
            let result_black = slot.result_black.expect("checked terminal slot");
            let game_index = slot.game_index;
            self.finished[game_index] = Some(FinishedGame {
                game_index,
                buffers: slot.buffers,
                result_black,
            });
            if self.next_game < self.game_count {
                let replacement =
                    GameSlot::new(self.next_game, &self.config).map_err(PyRuntimeError::new_err)?;
                self.slots[slot_index] = Some(replacement);
                self.next_game += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_plane_writer() {
        assert!(PlaneWriter::parse("unknown").is_err());
    }

    #[test]
    fn half_conversion_preserves_feature_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
    }
}
