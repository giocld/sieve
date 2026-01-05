/**
 * API client for Sieve backend
 */

const API_BASE = '/api';

export interface Player {
  player_name: string;
  PLAYER_ID: number;
  LEBRON: number;
  current_year_salary: number;
  value_gap: number;
  team: string;
  position?: string;
  [key: string]: unknown;
}

export interface Team {
  Abbrev: string;
  Team: string;
  WINS: number;
  LOSSES: number;
  Total_Payroll: number;
  Efficiency_Index: number;
  Logo_URL?: string;
  [key: string]: unknown;
}

export interface OverviewPlayer {
  name: string;
  team: string;
  value_gap: number;
  lebron: number;
  salary: number;
  player_id?: number;
}

export interface OverviewStats {
  season: string;
  num_players: number;
  num_teams: number;
  avg_salary_millions: number;
  avg_lebron: number;
  total_payroll_billions: number;
  top_value_player: string;
  top_value_gap: number;
  most_efficient_team: string;
  top_value_players: OverviewPlayer[];
  worst_value_players: OverviewPlayer[];
  top_performers: OverviewPlayer[];
}

export interface SeasonResponse {
  seasons: string[];
  current: string;
}

export interface PlayerFilters {
  season?: string;
  min_lebron?: number;
  min_salary?: number;
  max_salary?: number;
  search?: string;
}

export interface LineupFilters {
  team?: string | null;
  size?: number;
  min_minutes?: number;
  limit?: number;
}

export interface Lineup {
  GROUP_NAME: string;
  PLUS_MINUS: number;
  MIN: number;
  [key: string]: unknown;
}

export interface DiamondFinderPlayer {
  name: string;
  salary: number;
  lebron: number;
}

export interface ReplacementResult {
  player_name: string;
  PLAYER_ID?: number;
  salary: number;
  LEBRON: number;
  savings: number;
  match_score: number;
  archetype?: string;
  defense_role?: string;
  [key: string]: unknown;
}

// ============================================================================
// API Functions
// ============================================================================

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// Seasons
export async function getSeasons(): Promise<SeasonResponse> {
  return fetchJson<SeasonResponse>(`${API_BASE}/seasons`);
}

// Overview
export async function getOverview(season: string): Promise<OverviewStats> {
  return fetchJson<OverviewStats>(`${API_BASE}/overview?season=${season}`);
}

// Players
export async function getPlayers(filters: PlayerFilters = {}): Promise<{ players: Player[]; count: number; season: string }> {
  const params = new URLSearchParams();
  if (filters.season) params.set('season', filters.season);
  if (filters.min_lebron !== undefined) params.set('min_lebron', String(filters.min_lebron));
  if (filters.min_salary !== undefined) params.set('min_salary', String(filters.min_salary));
  if (filters.max_salary !== undefined) params.set('max_salary', String(filters.max_salary));
  if (filters.search) params.set('search', filters.search);

  return fetchJson(`${API_BASE}/players?${params}`);
}

// Teams
export async function getTeams(season: string): Promise<{ teams: Team[]; count: number; season: string }> {
  return fetchJson(`${API_BASE}/teams?season=${season}`);
}

// Charts (return Plotly JSON)
export async function getQuadrantChart(season: string): Promise<string> {
  const response = await fetch(`${API_BASE}/charts/quadrant?season=${season}`);
  return response.text();
}

export async function getTeamGridChart(season: string): Promise<string> {
  const response = await fetch(`${API_BASE}/charts/team-grid?season=${season}`);
  return response.text();
}

export async function getSalaryImpactChart(season: string, minLebron: number, minSalary: number, maxSalary: number): Promise<string> {
  const params = new URLSearchParams({
    season,
    min_lebron: String(minLebron),
    min_salary: String(minSalary),
    max_salary: String(maxSalary),
  });
  const response = await fetch(`${API_BASE}/charts/salary-impact?${params}`);
  return response.text();
}

export async function getUnderpaidChart(season: string, minLebron: number, minSalary: number, maxSalary: number): Promise<string> {
  const params = new URLSearchParams({
    season,
    min_lebron: String(minLebron),
    min_salary: String(minSalary),
    max_salary: String(maxSalary),
  });
  const response = await fetch(`${API_BASE}/charts/underpaid?${params}`);
  return response.text();
}

export async function getOverpaidChart(season: string, minLebron: number, minSalary: number, maxSalary: number): Promise<string> {
  const params = new URLSearchParams({
    season,
    min_lebron: String(minLebron),
    min_salary: String(minSalary),
    max_salary: String(maxSalary),
  });
  const response = await fetch(`${API_BASE}/charts/overpaid?${params}`);
  return response.text();
}

export async function getBeeswarmChart(season: string, minLebron: number, minSalary: number, maxSalary: number): Promise<string> {
  const params = new URLSearchParams({
    season,
    min_lebron: String(minLebron),
    min_salary: String(minSalary),
    max_salary: String(maxSalary),
  });
  const response = await fetch(`${API_BASE}/charts/beeswarm?${params}`);
  return response.text();
}

export async function getTeamRadarChart(team1: string, team2: string): Promise<string> {
  const response = await fetch(`${API_BASE}/charts/team-radar?team1=${team1}&team2=${team2}`);
  return response.text();
}

// Lineups
export async function getLineupTeams(): Promise<{ teams: Array<{ label: string; value: string }> }> {
  return fetchJson(`${API_BASE}/lineups/teams`);
}

export async function getBestLineups(filters: LineupFilters): Promise<{ lineups: Lineup[]; count: number }> {
  const params = new URLSearchParams();
  if (filters.team) params.set('team', filters.team);
  if (filters.size) params.set('size', String(filters.size));
  if (filters.min_minutes) params.set('min_minutes', String(filters.min_minutes));
  if (filters.limit) params.set('limit', String(filters.limit));

  return fetchJson(`${API_BASE}/lineups/best?${params}`);
}

export async function getWorstLineups(filters: LineupFilters): Promise<{ lineups: Lineup[]; count: number }> {
  const params = new URLSearchParams();
  if (filters.team) params.set('team', filters.team);
  if (filters.size) params.set('size', String(filters.size));
  if (filters.min_minutes) params.set('min_minutes', String(filters.min_minutes));
  if (filters.limit) params.set('limit', String(filters.limit));

  return fetchJson(`${API_BASE}/lineups/worst?${params}`);
}

export async function getBestLineupsChart(filters: LineupFilters): Promise<string> {
  const params = new URLSearchParams();
  if (filters.team) params.set('team', filters.team);
  if (filters.size) params.set('size', String(filters.size));
  if (filters.min_minutes) params.set('min_minutes', String(filters.min_minutes));

  const response = await fetch(`${API_BASE}/charts/lineups/best?${params}`);
  return response.text();
}

export async function getWorstLineupsChart(filters: LineupFilters): Promise<string> {
  const params = new URLSearchParams();
  if (filters.team) params.set('team', filters.team);
  if (filters.size) params.set('size', String(filters.size));
  if (filters.min_minutes) params.set('min_minutes', String(filters.min_minutes));

  const response = await fetch(`${API_BASE}/charts/lineups/worst?${params}`);
  return response.text();
}

export async function getLineupsScatterChart(filters: LineupFilters): Promise<string> {
  const params = new URLSearchParams();
  if (filters.team) params.set('team', filters.team);
  if (filters.size) params.set('size', String(filters.size));
  if (filters.min_minutes) params.set('min_minutes', String(filters.min_minutes));

  const response = await fetch(`${API_BASE}/charts/lineups/scatter?${params}`);
  return response.text();
}

// Similarity Engine
export async function getSimilarityPlayers(): Promise<{ players: string[] }> {
  return fetchJson(`${API_BASE}/similarity/players`);
}

export async function getPlayerSeasons(playerName: string): Promise<{ seasons: string[] }> {
  return fetchJson(`${API_BASE}/similarity/seasons/${encodeURIComponent(playerName)}`);
}

export async function findSimilarPlayers(player: string, season: string, excludeSelf: boolean = true): Promise<{
  similar: Array<{
    Player: string;
    Season: string;
    id: number;
    Stats: Record<string, number>;
    Position?: string;
    MatchScore?: number;
    Distance?: number;
  }>;
  target: {
    name: string;
    season: string;
    player_id: number | null;
    position: string;
  } | null;
}> {
  const params = new URLSearchParams({
    player,
    season,
    exclude_self: String(excludeSelf),
  });
  return fetchJson(`${API_BASE}/similarity/find?${params}`);
}

// Diamond Finder
export async function getDiamondFinderPlayers(season: string): Promise<{ players: DiamondFinderPlayer[] }> {
  return fetchJson(`${API_BASE}/diamond-finder/players?season=${season}`);
}

export async function findDiamondReplacements(player: string, season: string): Promise<{
  target: {
    name: string;
    salary: number;
    lebron: number;
    archetype: string;
    defense_role: string;
    player_id: number | null;
  };
  replacements: ReplacementResult[];
}> {
  const params = new URLSearchParams({ player, season });
  return fetchJson(`${API_BASE}/diamond-finder/find?${params}`);
}
