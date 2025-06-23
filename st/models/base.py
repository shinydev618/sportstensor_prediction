import os
import bittensor as bt
import aiohttp
import asyncio
import random
import numpy as np
import hashlib
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from common.data import MatchPrediction, League, ProbabilityChoice, get_league_from_string
from common.constants import LEAGUES_ALLOWING_DRAWS
from st.sport_prediction_model import SportPredictionModel

MINER_ENV_PATH = 'neurons/miner.env'
load_dotenv(dotenv_path=MINER_ENV_PATH)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError(f"ODDS_API_KEY not found in {MINER_ENV_PATH}")

API_URL = "https://api.the-odds-api.com/v4/sports/"

# Team name mappings for normalization
mismatch_teams_mapping = {
    "Orlando City SC": "Orlando City",
    "Inter Miami CF": "Inter Miami",
    "Atlanta United FC": "Atlanta United",
    "Montreal Impact": "CF Montréal",
    "D.C. United": "DC United",
    "Tottenham Hotspur": "Tottenham",
    "Columbus Crew SC": "Columbus Crew",
    "Minnesota United FC": "Minnesota United",
    "Vancouver Whitecaps FC": "Vancouver Whitecaps",
    "Leicester City": "Leicester",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "LA Galaxy": "L.A. Galaxy",
    "Oakland Athletics": "Athletics",
}

SPORTS_TYPES = [
    {
        'sport_key': 'baseball_mlb',
        'region': 'us,eu',
    },
    {
        'sport_key': 'americanfootball_nfl',
        'region': 'us,eu'
    },
    {
        'sport_key': 'soccer_usa_mls',
        'region': 'us,eu'
    },
    {
        'sport_key': 'soccer_epl',
        'region': 'uk,eu'
    },
    {
        'sport_key': 'basketball_nba',
        'region': 'us,eu'
    },
]

league_mapping = {
    'NBA': 'NBA',
    'NFL': 'NFL',
    'MLS': 'MLS',
    'EPL': 'EPL',
    'MLB': 'MLB',
}

class SportstensorBaseModel(SportPredictionModel):
    def __init__(self, prediction: MatchPrediction):
        super().__init__(prediction)
        self.boost_min_percent = 0.03
        self.boost_max_percent = 0.10
        self.probability_cap = 0.95
        self.max_retries = 3
        self.retry_delay = 0.5
        self.timeout = 3

    async def fetch_odds(self, sport_key: str, region: str) -> Optional[dict]:
        """Fetch odds from the new API."""
        url = f"{API_URL}{sport_key}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": region,
            "bookmakers": "pinnacle",
            "markets": "h2h"
        }
        async with aiohttp.ClientSession() as session:
            try:
                bt.logging.debug("Fetching odds from API...")
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        print(f"\n=== API Error ===\nStatus: {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"\n=== API Exception ===\n{str(e)}")
                return None

    def map_team_name(self, team_name: str) -> str:
        """Map team names using mismatch mapping."""
        return mismatch_teams_mapping.get(team_name, team_name)

    def odds_to_probabilities(self, home_odds: float, away_odds: float, draw_odds: Optional[float] = None) -> Dict[str, float]:
        """Convert odds to probabilities."""
        try:
            if home_odds is None or away_odds is None:
                print("Missing required odds values")
                return None

            # Convert odds to probabilities
            home_prob = 1 / home_odds if home_odds > 0 else 0
            away_prob = 1 / away_odds if away_odds > 0 else 0
            draw_prob = 1 / draw_odds if draw_odds and draw_odds > 0 else 0

            # Normalize probabilities
            total = home_prob + away_prob + draw_prob
            if total <= 0:
                print("Invalid odds values resulted in zero total probability")
                return None

            probabilities = {
                "home": home_prob / total,
                "away": away_prob / total,
            }

            if draw_odds:
                probabilities["draw"] = draw_prob / total
            
            return probabilities
        
        except Exception as e:
            print(f"Error converting odds to probabilities: {str(e)}")
            return None
        except Exception as e:
            bt.logging.error(f"Error converting odds to probabilities: {str(e)}")
            return None

    # ===== INTELLIGENT FALLBACK METHODS (NO FILES NEEDED) =====
    
    def get_team_strength_estimate(self, team_name: str, league: str) -> float:
        """Estimate team strength based on team name analysis (no external files)"""
        
        # Hash-based consistent strength (same team always gets same strength)
        team_hash = hashlib.md5(team_name.encode()).hexdigest()
        base_strength = int(team_hash[:8], 16) % 1000 / 1000.0
        
        # Known strong teams get strength bonus
        strong_teams_by_league = {
            'EPL': {
                'manchester city': 0.20, 'liverpool': 0.18, 'arsenal': 0.15,
                'chelsea': 0.15, 'tottenham': 0.12, 'manchester united': 0.12,
                'newcastle': 0.10, 'brighton': 0.08, 'west ham': 0.06
            },
            'NFL': {
                'kansas city chiefs': 0.18, 'buffalo bills': 0.15, 'philadelphia eagles': 0.15,
                'san francisco 49ers': 0.15, 'dallas cowboys': 0.12, 'green bay packers': 0.12,
                'baltimore ravens': 0.12, 'miami dolphins': 0.10
            },
            'NBA': {
                'boston celtics': 0.18, 'denver nuggets': 0.16, 'milwaukee bucks': 0.15,
                'phoenix suns': 0.15, 'golden state warriors': 0.14, 'miami heat': 0.12,
                'philadelphia 76ers': 0.12, 'los angeles lakers': 0.10
            },
            'MLB': {
                'houston astros': 0.15, 'los angeles dodgers': 0.15, 'atlanta braves': 0.14,
                'new york yankees': 0.12, 'toronto blue jays': 0.10, 'philadelphia phillies': 0.10,
                'san diego padres': 0.10, 'new york mets': 0.08
            },
            'MLS': {
                'lafc': 0.15, 'philadelphia union': 0.12, 'new york city fc': 0.12,
                'seattle sounders': 0.10, 'atlanta united': 0.10, 'la galaxy': 0.08,
                'portland timbers': 0.08, 'inter miami': 0.06
            }
        }
        
        team_lower = team_name.lower()
        strength_bonus = 0.0
        
        if league in strong_teams_by_league:
            # Check for exact matches first
            if team_lower in strong_teams_by_league[league]:
                strength_bonus = strong_teams_by_league[league][team_lower]
            else:
                # Check for partial matches
                for strong_team, bonus in strong_teams_by_league[league].items():
                    if any(word in team_lower for word in strong_team.split()):
                        strength_bonus = max(strength_bonus, bonus * 0.8)  # Slightly lower for partial match
        
        # Combine base strength with team recognition bonus
        final_strength = 0.6 * base_strength + 0.4 * (0.5 + strength_bonus)
        return np.clip(final_strength, 0.15, 0.85)

    def get_home_advantage_factor(self, league: str) -> float:
        """Get league-specific home advantage based on real sports statistics"""
        home_advantages = {
            'EPL': 0.08,      # ~8% home advantage in Premier League
            'MLS': 0.10,      # ~10% home advantage in MLS
            'NFL': 0.12,      # ~12% home advantage in NFL
            'NBA': 0.06,      # ~6% home advantage in NBA
            'MLB': 0.04,      # ~4% home advantage in MLB
        }
        return home_advantages.get(league, 0.07)  # Default 7%

    def intelligent_fallback_prediction(self, reason: str = "API failed"):
        """Make intelligent prediction when API fails (no external files needed)"""
        bt.logging.info(f"Using intelligent fallback prediction: {reason}")
        
        try:
            league_name = self.prediction.league.name if hasattr(self.prediction.league, 'name') else str(self.prediction.league)
            
            # Get team strengths
            home_strength = self.get_team_strength_estimate(self.prediction.homeTeamName, league_name)
            away_strength = self.get_team_strength_estimate(self.prediction.awayTeamName, league_name)
            
            # Get home advantage
            home_advantage = self.get_home_advantage_factor(league_name)
            
            # Calculate base probability
            strength_diff = home_strength - away_strength
            home_prob = 0.5 + (strength_diff * 0.4) + home_advantage
            
            # Add small controlled randomness (±3%)
            random_factor = (np.random.random() - 0.5) * 0.06
            home_prob += random_factor
            
            # Ensure reasonable bounds
            home_prob = np.clip(home_prob, 0.25, 0.75)
            away_prob = 1 - home_prob
            
            # Handle draws for leagues that allow them
            if self.prediction.league in LEAGUES_ALLOWING_DRAWS:
                # Reduce home/away probabilities for draw possibility
                draw_prob = 0.15 + np.random.random() * 0.10  # 15-25% draw probability
                home_prob *= (1 - draw_prob)
                away_prob *= (1 - draw_prob)
                
                max_prob = max(home_prob, away_prob, draw_prob)
                
                if max_prob == draw_prob:
                    self.prediction.probabilityChoice = ProbabilityChoice.DRAW
                    self.prediction.probability = draw_prob
                elif home_prob > away_prob:
                    self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                    self.prediction.probability = home_prob
                else:
                    self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                    self.prediction.probability = away_prob
            else:
                # No draws allowed
                if home_prob > away_prob:
                    self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                    self.prediction.probability = home_prob
                else:
                    self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                    self.prediction.probability = away_prob
            
            # Ensure minimum confidence
            if self.prediction.probability < 0.52:
                self.prediction.probability = 0.52
            
            bt.logging.info(
                f"Intelligent fallback: {self.prediction.probabilityChoice} "
                f"({self.prediction.probability:.3f}) - Home: {home_strength:.2f}, "
                f"Away: {away_strength:.2f}, Home Adv: {home_advantage:.2f}"
            )
            
        except Exception as e:
            bt.logging.error(f"Intelligent fallback failed: {e}")
            # Ultimate simple fallback
            self.simple_deterministic_fallback()

    def simple_deterministic_fallback(self):
        """Simple but deterministic fallback (better than pure random)"""
        bt.logging.warning("Using simple deterministic fallback")
        
        # Use team names for deterministic outcome
        home_hash = hash(self.prediction.homeTeamName)
        away_hash = hash(self.prediction.awayTeamName)
        
        if home_hash > away_hash:
            self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
            self.prediction.probability = 0.55
        else:
            self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
            self.prediction.probability = 0.55
    
    async def make_prediction(self):
        """Enhanced prediction with intelligent fallback."""
        bt.logging.info(f"Predicting {self.prediction.league} game...")
        
        try:
            # Convert the league to enum if it's not already one
            if not isinstance(self.prediction.league, League):
                try:
                    league_enum = get_league_from_string(str(self.prediction.league))
                    if league_enum is None:
                        bt.logging.error(f"Unknown league: {self.prediction.league}. Using intelligent fallback.")
                        self.intelligent_fallback_prediction("Unknown league")
                        return
                    self.prediction.league = league_enum
                except ValueError as e:
                    bt.logging.error(f"Failed to convert league: {self.prediction.league}. Error: {e}")
                    self.intelligent_fallback_prediction("League conversion failed")
                    return
            else:
                league_enum = self.prediction.league

            if not isinstance(self.prediction.league, League):
                bt.logging.error(f"Invalid league type: {type(self.prediction.league)}. Expected League enum.")
                self.intelligent_fallback_prediction("Invalid league type")
                return
            
            # Dynamically determine sport_key
            league_to_sport_key = {
                "NBA": "basketball_nba",
                "NFL": "americanfootball_nfl",
                "MLS": "soccer_usa_mls",
                "EPL": "soccer_epl",
                "MLB": "baseball_mlb",
                "English Premier League": "soccer_epl",
                "American Major League Soccer": "soccer_usa_mls",
            }

            league_key = self.prediction.league.name
            sport_key = league_to_sport_key.get(league_key)

            if not sport_key:
                bt.logging.error(f"Unknown league: {league_key}. Unable to determine sport_key.")
                self.intelligent_fallback_prediction(f"Unknown sport_key for {league_key}")
                return

            # Determine the region
            region = "us,eu" if sport_key in ["baseball_mlb", "americanfootball_nfl", "basketball_nba"] else "uk,eu"
            
            odds_data = await self.fetch_odds(sport_key, region)

            if not odds_data:
                bt.logging.error("No odds data fetched.")
                self.intelligent_fallback_prediction("No odds data from API")
                return

            # Find the match
            for odds in odds_data:
                home_team = self.map_team_name(self.prediction.homeTeamName)
                away_team = self.map_team_name(self.prediction.awayTeamName)

                if odds["home_team"] == home_team and odds["away_team"] == away_team:
                    bookmaker = next((b for b in odds["bookmakers"] if b["key"] == "pinnacle"), None)
                    if not bookmaker:
                        bt.logging.error("No Pinnacle odds found")
                        continue

                    market = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
                    if not market:
                        bt.logging.error("No h2h market found")
                        continue

                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    home_odds = outcomes.get(home_team)
                    away_odds = outcomes.get(away_team)
                    draw_odds = outcomes.get("Draw") if self.prediction.league in LEAGUES_ALLOWING_DRAWS else None

                    bt.logging.debug(f"Raw odds: {outcomes}")

                    if home_odds is None or away_odds is None:
                        bt.logging.error("Missing odds for one or both teams")
                        continue

                    probabilities = self.odds_to_probabilities(home_odds, away_odds, draw_odds)
                    bt.logging.debug(f"Calculated probabilities: {probabilities}")

                    if probabilities:
                        # Find the highest probability outcome
                        max_prob = max(probabilities["home"], probabilities["away"], probabilities.get("draw", 0))

                        if max_prob == probabilities["home"]:
                            self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                        elif max_prob == probabilities["away"]:
                            self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                        else:
                            self.prediction.probabilityChoice = ProbabilityChoice.DRAW

                        self.prediction.probability = max_prob
                        bt.logging.info(f"API prediction made: {self.prediction.probabilityChoice} with probability {self.prediction.probability}")
                        return

            # Match not found in API data - use intelligent fallback
            bt.logging.warning("Match not found in fetched odds data.")
            self.intelligent_fallback_prediction("Match not found in API data")
            return
            
        except Exception as e:
            bt.logging.error(f"Failed to make prediction: {str(e)}")
            self.intelligent_fallback_prediction(f"Exception: {str(e)}")
