/**
 * React Query hooks for Sieve API
 */

import { useQuery } from '@tanstack/react-query';
import * as api from '../lib/api';

// ============================================================================
// Query Keys (for cache invalidation)
// ============================================================================

export const queryKeys = {
  seasons: ['seasons'] as const,
  overview: (season: string) => ['overview', season] as const,
  players: (filters: api.PlayerFilters) => ['players', filters] as const,
  teams: (season: string) => ['teams', season] as const,

  // Charts
  quadrantChart: (season: string) => ['chart', 'quadrant', season] as const,
  teamGridChart: (season: string) => ['chart', 'team-grid', 'h-colorbar', season] as const,
  salaryImpactChart: (season: string, filters: { minLebron: number; maxLebron: number; minSalary: number; maxSalary: number }) =>
    ['chart', 'salary-impact', season, filters] as const,
  underpaidChart: (season: string, filters: { minLebron: number; maxLebron: number; minSalary: number; maxSalary: number }) =>
    ['chart', 'underpaid', season, filters] as const,
  overpaidChart: (season: string, filters: { minLebron: number; maxLebron: number; minSalary: number; maxSalary: number }) =>
    ['chart', 'overpaid', season, filters] as const,
  beeswarmChart: (season: string, filters: { minLebron: number; maxLebron: number; minSalary: number; maxSalary: number }) =>
    ['chart', 'beeswarm', season, filters] as const,
  teamRadarChart: (team1: string, team2: string) => ['chart', 'team-radar', team1, team2] as const,

  // Similarity
  similarityPlayers: ['similarity', 'players'] as const,
  playerSeasons: (player: string) => ['similarity', 'seasons', player] as const,
  similarPlayers: (player: string, season: string, excludeSelf: boolean) =>
    ['similarity', 'find', player, season, excludeSelf] as const,

  // Diamond Finder
  diamondFinderPlayers: (season: string) => ['diamond-finder', 'players', season] as const,
  diamondReplacements: (player: string, season: string) => ['diamond-finder', 'find', player, season] as const,
};

// ============================================================================
// Hooks
// ============================================================================

export function useSeasons() {
  return useQuery({
    queryKey: queryKeys.seasons,
    queryFn: api.getSeasons,
    staleTime: 1000 * 60 * 60, // 1 hour
  });
}

export function useOverview(season: string) {
  return useQuery({
    queryKey: queryKeys.overview(season),
    queryFn: () => api.getOverview(season),
    enabled: !!season,
  });
}

export function usePlayers(filters: api.PlayerFilters) {
  return useQuery({
    queryKey: queryKeys.players(filters),
    queryFn: () => api.getPlayers(filters),
    enabled: !!filters.season,
  });
}

export function useTeams(season: string) {
  return useQuery({
    queryKey: queryKeys.teams(season),
    queryFn: () => api.getTeams(season),
    enabled: !!season,
  });
}

// Chart Hooks
export function useQuadrantChart(season: string) {
  return useQuery({
    queryKey: queryKeys.quadrantChart(season),
    queryFn: () => api.getQuadrantChart(season),
    enabled: !!season,
  });
}

export function useTeamGridChart(season: string) {
  return useQuery({
    queryKey: queryKeys.teamGridChart(season),
    queryFn: () => api.getTeamGridChart(season),
    enabled: !!season,
  });
}

export function useSalaryImpactChart(season: string, minLebron: number, maxLebron: number, minSalary: number, maxSalary: number) {
  return useQuery({
    queryKey: queryKeys.salaryImpactChart(season, { minLebron, maxLebron, minSalary, maxSalary }),
    queryFn: () => api.getSalaryImpactChart(season, minLebron, maxLebron, minSalary, maxSalary),
    enabled: !!season,
  });
}

export function useUnderpaidChart(season: string, minLebron: number, maxLebron: number, minSalary: number, maxSalary: number) {
  return useQuery({
    queryKey: queryKeys.underpaidChart(season, { minLebron, maxLebron, minSalary, maxSalary }),
    queryFn: () => api.getUnderpaidChart(season, minLebron, maxLebron, minSalary, maxSalary),
    enabled: !!season,
  });
}

export function useOverpaidChart(season: string, minLebron: number, maxLebron: number, minSalary: number, maxSalary: number) {
  return useQuery({
    queryKey: queryKeys.overpaidChart(season, { minLebron, maxLebron, minSalary, maxSalary }),
    queryFn: () => api.getOverpaidChart(season, minLebron, maxLebron, minSalary, maxSalary),
    enabled: !!season,
  });
}

export function useBeeswarmChart(season: string, minLebron: number, maxLebron: number, minSalary: number, maxSalary: number) {
  return useQuery({
    queryKey: queryKeys.beeswarmChart(season, { minLebron, maxLebron, minSalary, maxSalary }),
    queryFn: () => api.getBeeswarmChart(season, minLebron, maxLebron, minSalary, maxSalary),
    enabled: !!season,
  });
}

export function useTeamRadarChart(team1: string, team2: string) {
  return useQuery({
    queryKey: queryKeys.teamRadarChart(team1, team2),
    queryFn: () => api.getTeamRadarChart(team1, team2),
    enabled: !!team1 && !!team2,
  });
}

// Similarity Hooks
export function useSimilarityPlayers() {
  return useQuery({
    queryKey: queryKeys.similarityPlayers,
    queryFn: api.getSimilarityPlayers,
  });
}

export function usePlayerSeasons(player: string) {
  return useQuery({
    queryKey: queryKeys.playerSeasons(player),
    queryFn: () => api.getPlayerSeasons(player),
    enabled: !!player,
  });
}

export function useSimilarPlayers(player: string, season: string, excludeSelf: boolean = true) {
  return useQuery({
    queryKey: queryKeys.similarPlayers(player, season, excludeSelf),
    queryFn: () => api.findSimilarPlayers(player, season, excludeSelf),
    enabled: !!player && !!season,
  });
}

// Diamond Finder Hooks
export function useDiamondFinderPlayers(season: string) {
  return useQuery({
    queryKey: queryKeys.diamondFinderPlayers(season),
    queryFn: () => api.getDiamondFinderPlayers(season),
    enabled: !!season,
  });
}

export function useDiamondReplacements(player: string, season: string) {
  return useQuery({
    queryKey: queryKeys.diamondReplacements(player, season),
    queryFn: () => api.findDiamondReplacements(player, season),
    enabled: !!player && !!season,
  });
}
