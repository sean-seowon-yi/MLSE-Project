# Data Documentation

This document provides an overview of the football data available in this project, sourced from two different providers: **SkillCorner** and **StatsBomb**.

## Overview

This project contains two distinct football datasets that complement each other:

- **SkillCorner**: Provides tracking data (player and ball positions) and physical metrics from broadcast video analysis
- **StatsBomb**: Provides detailed event-by-event data with tactical context

Both datasets use different approaches to capture football match information, making them suitable for different types of analysis.

---

## SkillCorner Data

### Source
SkillCorner Open Data - Broadcast tracking data collected through computer vision and machine learning from broadcast video.

### Dataset Coverage
- **Competition**: Australian A-League
- **Season**: 2024/2025
- **Number of Matches**: 10 matches
- **Data Type**: Broadcast tracking (10 frames per second)

### Directory Structure
```
SkillCorner/
├── data/
│   ├── matches.json                    # Match metadata index
│   ├── matches/                        # Individual match folders
│   │   └── {match_id}/
│   │       ├── {match_id}_match.json              # Match details and lineups
│   │       ├── {match_id}_tracking_extrapolated.jsonl  # Tracking data (10 fps)
│   │       ├── {match_id}_dynamic_events.csv      # Game Intelligence events
│   │       └── {match_id}_phases_of_play.csv      # Team possession phases
│   └── aggregates/
│       └── aus1league_physicalaggregates_20242025_midfielders.csv  # Season aggregates
```

### Data Files

#### 1. `matches.json`
**Format**: JSON array  
**Content**: Basic match information including:
- Match ID
- Date and time
- Home and away teams
- Competition and season IDs
- Match status

**Example**:
```json
{
  "id": 1886347,
  "date_time": "2024-11-30T04:00:00Z",
  "home_team": {"id": 4177, "short_name": "Auckland FC"},
  "away_team": {"id": 1805, "short_name": "Newcastle"},
  "status": "closed",
  "competition_id": 61,
  "season_id": 95
}
```

#### 2. `{match_id}_match.json`
**Format**: JSON object  
**Content**: Detailed match information including:
- Lineup information for both teams
- Player details (positions, playing time, cards, goals)
- Match periods (start/end frames, duration)
- Stadium information
- Pitch dimensions
- Team kits

**Key Fields**:
- `players`: Array of player objects with detailed stats
- `match_periods`: Period start/end frames and durations
- `pitch_length` and `pitch_width`: Field dimensions in meters
- `home_team_side`: Direction of play

#### 3. `{match_id}_tracking_extrapolated.jsonl`
**Format**: JSON Lines (one JSON object per line)  
**Frequency**: 10 frames per second  
**Content**: Frame-by-frame tracking data

**Structure per frame**:
```json
{
  "frame": 100,
  "timestamp": "00:00:09.00",
  "period": 1,
  "ball_data": {
    "x": 12.78,
    "y": 19.42,
    "z": 3.27,
    "is_detected": true
  },
  "possession": {
    "player_id": null,
    "group": "away team"
  },
  "image_corners_projection": {...},
  "player_data": [
    {
      "x": -41.03,
      "y": 4.64,
      "player_id": 51009,
      "is_detected": false
    },
    ...
  ]
}
```

**Coordinate System**:
- Origin (0,0) at center of pitch
- X-axis: Long side (length of field)
- Y-axis: Short side (width of field)
- Units: Meters
- `is_detected`: `true` if player/ball visible, `false` if extrapolated

**Player Tracking Coverage**:
- **Complete Coverage**: All 22 players (11 per team) are present in every frame during active match periods
- **Extrapolation**: When players are off-screen or occluded, their positions are estimated (extrapolated) based on previous tracking data and motion models. These positions are marked with `is_detected: false`
- **Pre-match Frames**: Some matches contain empty frames at the beginning (frames 0 to `start_frame-1`) representing pre-match footage before kickoff. The `match_periods` field in `{match_id}_match.json` indicates the actual `start_frame` and `end_frame` for each period
- **Consistency**: Across all 10 matches, once the match period begins (from `start_frame` onward), every frame contains all 22 players with either detected or extrapolated positions

#### 4. `{match_id}_dynamic_events.csv`
**Format**: CSV  
**Content**: Game Intelligence dynamic events with extensive attributes

**Event Categories**:
- **Player Possession** (event_type_id: 8): When a player has the ball
- **Passing Options** (event_type_id: 7): Available passing options
- **Off-ball Runs** (event_type_id: 1): Player movements without the ball
- **On-ball Engagements** (event_type_id: 9): Defensive actions (pressing, pressure, etc.)

**Key Columns**:
- `event_id`: Unique identifier (unique per game)
- `frame_start`, `frame_end`: Frame range of the event
- `time_start`, `time_end`: Timestamp range
- `player_id`, `player_name`: Player involved
- `x_start`, `y_start`, `x_end`, `y_end`: Spatial coordinates
- `event_type`, `event_subtype`: Type of event
- `pass_outcome`: For passes (successful/unsuccessful)
- `team_in_possession_phase_type`: Tactical phase (build_up, create, direct, finish, chaotic)
- `team_out_of_possession_phase_type`: Defensive phase (high_block, medium_block, low_block, etc.)
- Many advanced metrics: xThreat, xPass completion, passing options, defensive line breaks, etc.

**Note**: X/Y coordinates in this file are NOT scaled to standard pitch size and may require adjustment.

#### 5. `{match_id}_phases_of_play.csv`
**Format**: CSV  
**Content**: Team possession phases

**Key Columns**:
- `frame_start`, `frame_end`: Phase boundaries
- `team_in_possession_id`: Team with possession
- `team_in_possession_phase_type`: Phase type (build_up, create, direct, finish, chaotic)
- `team_out_of_possession_phase_type`: Opposing team's defensive phase
- `n_player_possessions_in_phase`: Number of player possessions
- `team_possession_loss_in_phase`: Whether possession was lost
- `team_possession_lead_to_goal`: Whether phase led to a goal
- `team_possession_lead_to_shot`: Whether phase led to a shot
- Spatial metrics: width, length, positions

**Phase Types**:
- **build_up**: Building from the back
- **create**: Creating chances
- **direct**: Direct attacking play
- **finish**: Finishing phase
- **chaotic**: Unstructured play

#### 6. `aus1league_physicalaggregates_20242025_midfielders.csv`
**Format**: CSV  
**Content**: Season-level aggregated physical performance metrics for midfielders

**Key Metrics**:
- `total_distance_full_all`: Total distance covered (meters)
- `total_metersperminute_full_all`: Average meters per minute
- `running_distance_full_all`: Distance in running zones
- `hsr_distance_full_all`: High-speed running distance
- `sprint_distance_full_all`: Sprint distance
- `hi_distance_full_all`: High-intensity distance
- Acceleration/deceleration counts
- Metrics split by TIP (Team in Possession) and OTIP (Out of Possession)

**Filter**: Only includes performances above 60 minutes

### Limitations
- ~97% accuracy in player identification
- Some data points may be erroneous
- Speed/acceleration smoothing recommended for raw data
- X/Y coordinates in dynamic events require scaling adjustment

---

## StatsBomb Data

### Source
StatsBomb Open Data - Detailed event data collected through manual annotation and analysis.

### Dataset Coverage
- **Multiple Competitions**: La Liga, Premier League, Champions League, World Cup, Women's World Cup, and more
- **Multiple Seasons**: Various seasons from different competitions
- **Data Type**: Event-by-event data with tactical context

### Directory Structure
```
StatsBomb/
├── data/
│   ├── competitions.json              # Available competitions and seasons
│   ├── matches/                       # Match metadata by competition
│   │   └── {competition_id}/
│   │       └── {season_id}.json      # Matches for that season
│   ├── events/                        # Event data
│   │   └── {match_id}.json           # All events for a match
│   ├── lineups/                       # Lineup data
│   │   └── {match_id}.json           # Lineups for a match
│   └── three-sixty/                   # 360 data (spatial context)
│       └── {match_id}.json           # 360 frames for selected matches
```

### Data Files

#### 1. `competitions.json`
**Format**: JSON array  
**Content**: List of available competitions and seasons

**Example**:
```json
{
  "competition_id": 11,
  "season_id": 1,
  "country_name": "Spain",
  "competition_name": "La Liga",
  "competition_gender": "male",
  "season_name": "2017/2018",
  "match_available": "2024-02-13T02:35:28.134882"
}
```

#### 2. `matches/{competition_id}/{season_id}.json`
**Format**: JSON array  
**Content**: Match metadata for a specific competition and season

**Key Fields**:
- `match_id`: Unique match identifier
- `match_date`: Date of the match
- `kick_off`: Kick-off time
- `home_team`, `away_team`: Team information including managers
- `home_score`, `away_score`: Final scores
- `stadium`: Stadium information
- `referee`: Referee information
- `competition_stage`: Stage of competition (Regular Season, etc.)
- `match_week`: Week number in season

#### 3. `events/{match_id}.json`
**Format**: JSON array  
**Content**: All events that occurred during the match, in chronological order

**Event Structure**:
```json
{
  "id": "unique-event-id",
  "index": 1,
  "period": 1,
  "timestamp": "00:00:00.000",
  "minute": 0,
  "second": 0,
  "type": {
    "id": 35,
    "name": "Starting XI"
  },
  "possession": 1,
  "possession_team": {
    "id": 217,
    "name": "Barcelona"
  },
  "play_pattern": {
    "id": 1,
    "name": "Regular Play"
  },
  "team": {
    "id": 217,
    "name": "Barcelona"
  },
  "location": [x, y],
  "player": {...},
  "position": {...},
  ...
}
```

**Common Event Types**:
- **Starting XI** (35): Lineup announcement
- **Pass** (30): Pass events
- **Shot** (16): Shot attempts
- **Goal** (16 with outcome): Goals scored
- **Ball Receipt** (14): Receiving the ball
- **Carry** (43): Ball carrying
- **Pressure** (27): Pressing actions
- **Duel** (50): 1v1 situations
- **Foul Committed** (4): Fouls
- **Substitution** (19): Player substitutions
- And many more...

**Event Attributes** (varies by event type):
- `location`: [x, y] coordinates (0-120 for x, 0-80 for y)
- `player`: Player who performed the action
- `outcome`: Result of the action
- `technique`: Technique used
- `body_part`: Body part used
- `pass`: Pass-specific details (length, angle, recipient, etc.)
- `shot`: Shot-specific details (end_location, outcome, technique, etc.)
- `tactics`: Formation and lineup at time of event

#### 4. `lineups/{match_id}.json`
**Format**: JSON array  
**Content**: Lineup information for both teams

**Structure**:
```json
{
  "team_id": 217,
  "team_name": "Barcelona",
  "lineup": [
    {
      "player_id": 3501,
      "player_name": "Philippe Coutinho Correia",
      "player_nickname": "Philippe Coutinho",
      "jersey_number": 14,
      "country": {...},
      "cards": [],
      "positions": [
        {
          "position_id": 12,
          "position": "Right Midfield",
          "from": "00:00",
          "to": "78:15",
          "from_period": 1,
          "to_period": 2,
          "start_reason": "Starting XI",
          "end_reason": "Substitution - Off (Tactical)"
        }
      ]
    },
    ...
  ]
}
```

**Key Information**:
- Player details (name, nickname, jersey number, country)
- Position history (where and when they played)
- Cards received
- Substitution information

#### 5. `three-sixty/{match_id}.json` (if available)
**Format**: JSON array  
**Content**: Spatial context data for selected events (freeze frames: positions of other players at each event moment).

**Note**: Not all matches have 360 data available. Check `match_available_360` in competitions.json. The **GNN_StatsBomb** pipeline uses 360 by default: only matches with a `three-sixty/{match_id}.json` file are loaded, and only events that have a 360 frame are kept, so that player similarity can use spatial context (teammates/opponents around the ball).

### Coordinate System
- **X-axis**: 0-120 (length of field)
- **Y-axis**: 0-80 (width of field)
- Origin at bottom-left corner (from attacking perspective)
- Standardized pitch dimensions

### Key Features
- **Rich Event Context**: Each event includes tactical formation, play pattern, and possession information
- **Detailed Pass Data**: Pass length, angle, height, recipient, outcome
- **Shot Analysis**: Shot location, end location, technique, outcome, goalkeeper position
- **Tactical Information**: Formation changes, player positions over time
- **Multiple Competitions**: Wide variety of leagues and tournaments

---

## FIFA / EA Sports FC Data

### Source
EA Sports FC (formerly FIFA) player attribute data. These files are downloaded externally and are not included in the repository.

### Purpose
Used to **validate** the GNN player similarity system by comparing the FIFA stats of a query player and their GNN-recommended substitute. If the GNN correctly identifies similar players, their FIFA attribute profiles should also be similar.

### Directory Structure
```
FIFA_data/
├── male_players.csv                       # Raw FIFA male player data (downloaded)
├── female_players.csv                     # Raw FIFA female player data (downloaded)
├── statsbomb_male_players_fifa.csv        # Matched: StatsBomb male players → FIFA stats
├── statsbomb_female_players_fifa.csv      # Matched: StatsBomb female players → FIFA stats
├── unmatched_players.txt                  # StatsBomb 360 players not found in FIFA data
├── test_fifa_comparison.txt               # Text report from comparison test
├── radar_comparison.png                   # Query vs substitute stat profile radar charts
├── similarity_vs_stat_diff.png            # Cosine similarity vs FIFA stat difference scatter
├── stat_difference_breakdown.png          # Per-attribute mean absolute difference bars
└── evaluation_summary.png                 # Position match rate, rating gap, agreement boxplots
```

### Matching Process

`GNN_StatsBomb/match_fifa_players.py` links StatsBomb 360 players to FIFA data using:

- **Name normalization**: Unicode NFKD decomposition, accent stripping, hyphen/space handling.
- **Country aliases**: Maps variations (e.g. "Korea Republic" ↔ "Korea DPR" ↔ "South Korea").
- **Competition-to-FIFA-version mapping**: Aligns StatsBomb season years to the correct FIFA game release.
- **Multi-stage fuzzy matching**: Exact match → subsequence → SequenceMatcher with configurable thresholds.
- **Post-match deduplication**: Ensures one-to-one mapping.

Outputs: `statsbomb_male_players_fifa.csv`, `statsbomb_female_players_fifa.csv`, `unmatched_players.txt`.

### Key FIFA Attributes

| Group | Attributes |
|-------|-----------|
| **Main 6** | Pace, Shooting, Passing, Dribbling, Defending, Physical |
| **Attacking** | Crossing, Finishing, Heading Accuracy, Short Passing, Volleys |
| **Skill** | Dribbling, Curve, FK Accuracy, Long Passing, Ball Control |
| **Movement** | Acceleration, Sprint Speed, Agility, Reactions, Balance |
| **Power** | Shot Power, Jumping, Stamina, Strength, Long Shots |
| **Mentality** | Aggression, Interceptions, Positioning, Vision, Penalties, Composure |
| **Defending** | Marking Awareness, Standing Tackle, Sliding Tackle |
| **Goalkeeping** | Diving, Handling, Kicking, Positioning, Reflexes |

---

## Data Comparison

| Feature | SkillCorner | StatsBomb |
|---------|-------------|-----------|
| **Data Type** | Tracking (positions) | Events (actions) |
| **Frequency** | 10 fps continuous | Event-based |
| **Coverage** | 10 matches (A-League) | Multiple competitions |
| **Physical Metrics** | Yes (aggregated) | No |
| **Tactical Phases** | Yes (phases of play) | Yes (formation/tactics) |
| **Passing Options** | Yes (dynamic events) | No |
| **Off-ball Runs** | Yes | Limited |
| **Coordinate System** | Meters, center origin | 0-120/0-80, corner origin |
| **Player Identification** | ~97% accuracy | Manual annotation |

---

## Use Cases

### SkillCorner Data
- **Physical Performance Analysis**: Distance covered, speed zones, accelerations
- **Tactical Analysis**: Team shapes, pressing intensity, defensive lines
- **Player Movement**: Off-ball runs, positioning, spacing
- **Possession Analysis**: Phases of play, build-up patterns
- **Passing Analysis**: Available options, line breaks, xThreat

### StatsBomb Data
- **Event Analysis**: Pass completion, shot quality, defensive actions
- **Tactical Analysis**: Formations, player roles, team structures
- **Performance Metrics**: Goals, assists, key passes, defensive actions
- **Match Context**: Score, time, game state
- **Comparative Analysis**: Across multiple leagues and seasons

---

## Data Integration

These datasets can be combined for comprehensive analysis:

1. **Match Linking**: Use match dates and teams to link SkillCorner tracking with StatsBomb events
2. **Temporal Alignment**: Align tracking frames with event timestamps
3. **Spatial Alignment**: Convert between coordinate systems for unified analysis
4. **Complementary Analysis**: Use tracking for movement patterns and StatsBomb for event outcomes

---

## Additional Resources

### SkillCorner
- Documentation: See `SkillCorner/README.md`
- Tutorials: Available in `SkillCorner/resources/Tutorials/`
- Dynamic Events Spec: Referenced in README
- Phases of Play Spec: Referenced in README

### StatsBomb
- Documentation: Available in `StatsBomb/doc/` directory
- Open Data Specification: `StatsBomb Open Data Specification v1.1.pdf`
- Event Documentation: `Open Data Events v4.0.0.pdf`
- Lineup Documentation: `Open Data Lineups v2.0.0.pdf`
- 360 Documentation: `Open Data 360 Frames v1.0.0.pdf`

---

## Notes

- **Coordinate Systems**: Be aware of different coordinate systems when combining data
- **Data Quality**: SkillCorner tracking has ~97% player identification accuracy
- **Missing Data**: Some matches may not have all data types (e.g., 360 data in StatsBomb)
- **Scaling**: SkillCorner dynamic events coordinates may need adjustment for standard pitch size
- **Temporal Precision**: SkillCorner provides 10 fps tracking; StatsBomb provides event timestamps

---

## Credits

- **SkillCorner**: Data provided by SkillCorner in collaboration with PySport
- **StatsBomb**: StatsBomb Open Data - freely available for research and analysis

Please credit the respective data providers when using this data in publications or analysis.
