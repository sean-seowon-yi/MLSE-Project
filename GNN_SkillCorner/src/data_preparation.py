"""
Phase 1: Data Preparation

This module handles:
1. Loading SkillCorner tracking data
2. Filtering for active play frames
3. Team-aware normalization (crucial for preserving left/right sidedness)
4. Feature engineering for player vectors

The normalization strategy ensures that:
- All teams appear to attack towards positive X (right side)
- "Left" is always negative Y, "Right" is always positive Y
- This preserves tactical role distinction (Left Back vs Right Back)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
from tqdm import tqdm

from .config import DataConfig, get_config


@dataclass
class PlayerFrame:
    """Data structure for a single player in a single frame."""
    player_id: int
    team_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    is_sprinting: bool
    is_home_team: bool
    role_name: str
    trackable_object: int


@dataclass 
class FrameData:
    """Data structure for a complete frame."""
    match_id: int
    frame_number: int
    timestamp: str
    period: int
    players: List[PlayerFrame]
    ball_x: float
    ball_y: float
    home_attacking_direction: str  # 'left_to_right' or 'right_to_left'


class SkillCornerDataLoader:
    """
    Loads and preprocesses SkillCorner tracking data.
    
    Key responsibilities:
    - Load match metadata and tracking data
    - Filter for valid frames (22 players, ball in play)
    - Apply team-aware coordinate normalization
    - Compute velocity and sprint features
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.base_path = Path(config.skillcorner_base_path)
        self.matches_info = self._load_matches_index()
        
    def _load_matches_index(self) -> List[Dict]:
        """Load the matches.json index file."""
        matches_file = self.base_path / self.config.matches_file
        with open(matches_file, 'r') as f:
            return json.load(f)
    
    def _load_match_metadata(self, match_id: int) -> Dict:
        """Load detailed match metadata including player info and pitch dimensions."""
        match_file = self.base_path / "matches" / str(match_id) / f"{match_id}_match.json"
        with open(match_file, 'r') as f:
            return json.load(f)
    
    def _load_tracking_data(self, match_id: int) -> Generator[Dict, None, None]:
        """Load tracking data frame by frame (memory efficient generator)."""
        tracking_file = self.base_path / "matches" / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
        with open(tracking_file, 'r') as f:
            for line in f:
                yield json.load(line) if line.strip() else None
    
    def _get_player_info(self, match_metadata: Dict) -> Dict[int, Dict]:
        """
        Extract player information mapping player_id to player details.
        
        Note: The tracking data's 'player_id' field actually contains the player's 'id'
        from the metadata, NOT the 'trackable_object'. This mapping uses 'id' as the key.
        
        Returns dict: player_id -> {player_id, team_id, role_name, is_home_team, trackable_object}
        """
        player_map = {}
        home_team_id = match_metadata['home_team']['id']
        
        for player in match_metadata.get('players', []):
            player_id = player.get('id')
            if player_id:
                player_map[player_id] = {
                    'player_id': player['id'],
                    'team_id': player['team_id'],
                    'role_name': player.get('player_role', {}).get('name', 'Unknown'),
                    'is_home_team': player['team_id'] == home_team_id,
                    'trackable_object': player.get('trackable_object')
                }
        
        return player_map
    
    def _get_attacking_direction(self, match_metadata: Dict, period: int) -> str:
        """
        Determine home team's attacking direction for the given period.
        
        Returns: 'left_to_right' or 'right_to_left'
        """
        home_team_side = match_metadata.get('home_team_side', ['right_to_left', 'left_to_right'])
        # home_team_side[0] is period 1, home_team_side[1] is period 2
        if period <= len(home_team_side):
            return home_team_side[period - 1]
        return home_team_side[-1]  # Default to last known direction
    
    def _normalize_coordinates(
        self, 
        x: float, 
        y: float, 
        is_home_team: bool,
        home_attacking_direction: str,
        pitch_length: float,
        pitch_width: float
    ) -> Tuple[float, float]:
        """
        Apply team-aware normalization to preserve left/right sidedness.
        
        The Rule: Rotate data so every team appears to attack towards Right (Positive X).
        
        Logic:
        - If team is attacking right: Keep coordinates as they are
        - If team is attacking left: Multiply x and y by -1
        
        This ensures:
        - "Left" is always Negative Y
        - "Right" is always Positive Y
        - Model can distinguish Left Back from Right Back
        
        Args:
            x, y: Original coordinates (meters, center origin)
            is_home_team: Whether player belongs to home team
            home_attacking_direction: Home team's attacking direction for this period
            pitch_length, pitch_width: Pitch dimensions
            
        Returns:
            Normalized (x, y) coordinates
        """
        # Determine if this player's team is attacking left
        if is_home_team:
            attacking_left = 'right_to_left' in home_attacking_direction
        else:
            # Away team attacks opposite direction
            attacking_left = 'left_to_right' in home_attacking_direction
        
        # Apply rotation if attacking left
        if attacking_left:
            x = -x
            y = -y
        
        # Optionally normalize to [-1, 1] range
        if self.config.normalize_to_unit:
            x = x / (pitch_length / 2)
            y = y / (pitch_width / 2)
            # Clip to handle any edge cases
            x = np.clip(x, -1.0, 1.0)
            y = np.clip(y, -1.0, 1.0)
        
        return x, y
    
    def _compute_velocity(
        self,
        current_x: float,
        current_y: float,
        prev_x: Optional[float],
        prev_y: Optional[float],
        dt: float = 0.1  # 10 fps = 0.1 seconds per frame
    ) -> Tuple[float, float, float]:
        """
        Compute velocity components and speed.
        
        Args:
            current_x, current_y: Current position
            prev_x, prev_y: Previous position (None if first frame)
            dt: Time delta between frames
            
        Returns:
            (vx, vy, speed) - velocity components and magnitude
        """
        if prev_x is None or prev_y is None:
            return 0.0, 0.0, 0.0
        
        vx = (current_x - prev_x) / dt
        vy = (current_y - prev_y) / dt
        speed = np.sqrt(vx**2 + vy**2)
        
        return vx, vy, speed
    
    def _is_sprinting(self, speed: float) -> bool:
        """Check if player speed exceeds sprint threshold."""
        return speed > self.config.sprint_threshold
    
    def process_match(self, match_id: int) -> List[FrameData]:
        """
        Process a single match and return list of valid frame data.
        
        This is the main entry point for match processing:
        1. Load metadata and tracking data
        2. Filter for valid frames
        3. Apply normalization
        4. Compute velocity features
        
        Args:
            match_id: SkillCorner match ID
            
        Returns:
            List of FrameData objects, sampled at configured rate
        """
        print(f"Processing match {match_id}...")
        
        # Load metadata
        metadata = self._load_match_metadata(match_id)
        player_info = self._get_player_info(metadata)
        pitch_length = metadata.get('pitch_length', self.config.pitch_length)
        pitch_width = metadata.get('pitch_width', self.config.pitch_width)
        
        # Get match periods for filtering
        match_periods = metadata.get('match_periods', [])
        period_ranges = {}
        for period in match_periods:
            period_num = period['period']
            period_ranges[period_num] = (period['start_frame'], period['end_frame'])
        
        # Track previous positions for velocity calculation
        prev_positions = {}  # trackable_object -> (x, y)
        
        # Process frames
        processed_frames = []
        frame_count = 0
        
        tracking_file = self.base_path / "matches" / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
        
        with open(tracking_file, 'r') as f:
            for line in tqdm(f, desc=f"Match {match_id}", leave=False):
                if not line.strip():
                    continue
                    
                frame = json.loads(line)
                frame_num = frame.get('frame', 0)
                period = frame.get('period', 0)
                
                # Skip frames outside match periods
                if period not in period_ranges:
                    continue
                start_frame, end_frame = period_ranges[period]
                if frame_num < start_frame or frame_num > end_frame:
                    continue
                
                # Sample at configured rate
                if frame_num % self.config.sampling_rate != 0:
                    # Still update previous positions for velocity
                    for pdata in frame.get('player_data', []):
                        pid = pdata.get('player_id')  # This is actually player 'id' in the metadata
                        if pid and pdata.get('x') is not None:
                            prev_positions[pid] = (pdata['x'], pdata['y'])
                    continue
                
                # Get player data
                player_data = frame.get('player_data', [])
                
                # Filter: require exactly 22 players
                valid_players = [p for p in player_data if p.get('x') is not None]
                if len(valid_players) != 22:
                    continue
                
                # Get attacking direction for this period
                home_attacking_dir = self._get_attacking_direction(metadata, period)
                
                # Get ball data and normalize to same coordinate system as players
                # (team-aware: flip when home attacks left so ball is in "attack right" space)
                ball_data = frame.get('ball_data', {})
                raw_ball_x = ball_data.get('x', 0.0)
                raw_ball_y = ball_data.get('y', 0.0)
                ball_x, ball_y = self._normalize_coordinates(
                    raw_ball_x, raw_ball_y,
                    is_home_team=True,  # Use home perspective for ball (consistent ref)
                    home_attacking_direction=home_attacking_dir,
                    pitch_length=pitch_length,
                    pitch_width=pitch_width
                )
                
                # Process each player
                players = []
                for pdata in valid_players:
                    # Note: 'player_id' in tracking data is actually the player's 'id' from metadata
                    player_id_in_tracking = pdata.get('player_id')
                    
                    # Get player info
                    info = player_info.get(player_id_in_tracking, {})
                    if not info:
                        continue
                    
                    # Raw coordinates
                    raw_x = pdata['x']
                    raw_y = pdata['y']
                    
                    # Normalize coordinates
                    norm_x, norm_y = self._normalize_coordinates(
                        raw_x, raw_y,
                        info['is_home_team'],
                        home_attacking_dir,
                        pitch_length, pitch_width
                    )
                    
                    # Get previous position (also normalized)
                    prev_pos = prev_positions.get(player_id_in_tracking)
                    if prev_pos:
                        prev_norm_x, prev_norm_y = self._normalize_coordinates(
                            prev_pos[0], prev_pos[1],
                            info['is_home_team'],
                            home_attacking_dir,
                            pitch_length, pitch_width
                        )
                    else:
                        prev_norm_x, prev_norm_y = None, None
                    
                    # Compute velocity (normalized coords for vx, vy in graph)
                    vx, vy, _ = self._compute_velocity(
                        norm_x, norm_y, prev_norm_x, prev_norm_y
                    )
                    # Speed in m/s for sprint flag (doc: sprint_threshold 7.0 m/s)
                    if prev_pos is not None:
                        raw_dx = raw_x - prev_pos[0]
                        raw_dy = raw_y - prev_pos[1]
                        speed_ms = np.sqrt(raw_dx**2 + raw_dy**2) / 0.1
                    else:
                        speed_ms = 0.0
                    
                    # Create player frame
                    player_frame = PlayerFrame(
                        player_id=info['player_id'],
                        team_id=info['team_id'],
                        x=norm_x,
                        y=norm_y,
                        vx=vx,
                        vy=vy,
                        speed=speed_ms,
                        is_sprinting=self._is_sprinting(speed_ms),
                        is_home_team=info['is_home_team'],
                        role_name=info['role_name'],
                        trackable_object=info.get('trackable_object', player_id_in_tracking)
                    )
                    players.append(player_frame)
                    
                    # Update previous position
                    prev_positions[player_id_in_tracking] = (raw_x, raw_y)
                
                # Only add if we have exactly 22 players
                if len(players) == 22:
                    frame_data = FrameData(
                        match_id=match_id,
                        frame_number=frame_num,
                        timestamp=frame.get('timestamp', ''),
                        period=period,
                        players=players,
                        ball_x=ball_x,
                        ball_y=ball_y,
                        home_attacking_direction=home_attacking_dir
                    )
                    processed_frames.append(frame_data)
                    frame_count += 1
        
        print(f"  Processed {frame_count} valid frames from match {match_id}")
        return processed_frames
    
    def process_all_matches(self) -> List[FrameData]:
        """Process all matches in the dataset."""
        all_frames = []
        
        for match_info in tqdm(self.matches_info, desc="Processing matches"):
            match_id = match_info['id']
            try:
                frames = self.process_match(match_id)
                all_frames.extend(frames)
            except Exception as e:
                print(f"Error processing match {match_id}: {e}")
                continue
        
        print(f"\nTotal processed frames: {len(all_frames)}")
        return all_frames
    
    def save_processed_data(self, frames: List[FrameData], output_path: Optional[str] = None):
        """Save processed frames to disk."""
        if output_path is None:
            output_path = Path(self.config.output_dir) / "processed_frames.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = []
        for frame in frames:
            frame_dict = {
                'match_id': frame.match_id,
                'frame_number': frame.frame_number,
                'timestamp': frame.timestamp,
                'period': frame.period,
                'ball_x': frame.ball_x,
                'ball_y': frame.ball_y,
                'home_attacking_direction': frame.home_attacking_direction,
                'players': [
                    {
                        'player_id': p.player_id,
                        'team_id': p.team_id,
                        'x': p.x,
                        'y': p.y,
                        'vx': p.vx,
                        'vy': p.vy,
                        'speed': p.speed,
                        'is_sprinting': p.is_sprinting,
                        'is_home_team': p.is_home_team,
                        'role_name': p.role_name,
                        'trackable_object': p.trackable_object
                    }
                    for p in frame.players
                ]
            }
            data.append(frame_dict)
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {len(data)} frames to {output_path}")
    
    @staticmethod
    def load_processed_data(path: str) -> List[FrameData]:
        """Load processed frames from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        frames = []
        for frame_dict in data:
            players = [
                PlayerFrame(
                    player_id=p['player_id'],
                    team_id=p['team_id'],
                    x=p['x'],
                    y=p['y'],
                    vx=p['vx'],
                    vy=p['vy'],
                    speed=p['speed'],
                    is_sprinting=p['is_sprinting'],
                    is_home_team=p['is_home_team'],
                    role_name=p['role_name'],
                    trackable_object=p['trackable_object']
                )
                for p in frame_dict['players']
            ]
            frame = FrameData(
                match_id=frame_dict['match_id'],
                frame_number=frame_dict['frame_number'],
                timestamp=frame_dict['timestamp'],
                period=frame_dict['period'],
                players=players,
                ball_x=frame_dict['ball_x'],
                ball_y=frame_dict['ball_y'],
                home_attacking_direction=frame_dict['home_attacking_direction']
            )
            frames.append(frame)
        
        return frames


def get_player_metadata(frames: List[FrameData]) -> pd.DataFrame:
    """
    Extract unique player metadata from processed frames.
    
    Returns DataFrame with columns: player_id, team_id, role_name, is_home_team
    """
    player_info = {}
    
    for frame in frames:
        for player in frame.players:
            if player.player_id not in player_info:
                player_info[player.player_id] = {
                    'player_id': player.player_id,
                    'team_id': player.team_id,
                    'role_name': player.role_name,
                    'is_home_team': player.is_home_team
                }
    
    return pd.DataFrame(list(player_info.values()))


if __name__ == "__main__":
    # Test data preparation pipeline
    config = get_config()
    loader = SkillCornerDataLoader(config.data)
    
    # Process first match as test
    match_id = loader.matches_info[0]['id']
    frames = loader.process_match(match_id)
    
    print(f"\n=== Sample Frame ===")
    if frames:
        sample = frames[0]
        print(f"Match: {sample.match_id}, Frame: {sample.frame_number}")
        print(f"Period: {sample.period}, Attacking direction: {sample.home_attacking_direction}")
        print(f"Number of players: {len(sample.players)}")
        print(f"\nSample player:")
        p = sample.players[0]
        print(f"  ID: {p.player_id}, Role: {p.role_name}")
        print(f"  Position: ({p.x:.3f}, {p.y:.3f})")
        print(f"  Velocity: ({p.vx:.3f}, {p.vy:.3f}), Speed: {p.speed:.3f}")
        print(f"  Sprinting: {p.is_sprinting}, Home team: {p.is_home_team}")
