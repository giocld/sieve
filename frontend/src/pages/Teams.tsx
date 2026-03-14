/**
 * teams page - clean trading dashboard style
 */

import { useState, useMemo } from 'react';
import { useTeams, useQuadrantChart, useTeamGridChart, useTeamRadarChart } from '../hooks/useApi';
import type { Team } from '../lib/api';
import { PageHeader, Panel, PanelHeader, PanelBody, ChartPanel, ChartGrid, DataTable } from '../components';
import { Select } from '../components/FormControls';
import { PageLoading, ErrorDisplay } from '../components/Loading';

// team abbreviation to full name mapping
const TEAM_NAMES: Record<string, string> = {
  ATL: 'Atlanta Hawks', BOS: 'Boston Celtics', BKN: 'Brooklyn Nets', CHA: 'Charlotte Hornets',
  CHI: 'Chicago Bulls', CLE: 'Cleveland Cavaliers', DAL: 'Dallas Mavericks', DEN: 'Denver Nuggets',
  DET: 'Detroit Pistons', GSW: 'Golden State Warriors', HOU: 'Houston Rockets', IND: 'Indiana Pacers',
  LAC: 'Los Angeles Clippers', LAL: 'Los Angeles Lakers', MEM: 'Memphis Grizzlies', MIA: 'Miami Heat',
  MIL: 'Milwaukee Bucks', MIN: 'Minnesota Timberwolves', NOP: 'New Orleans Pelicans', NYK: 'New York Knicks',
  OKC: 'Oklahoma City Thunder', ORL: 'Orlando Magic', PHI: 'Philadelphia 76ers', PHX: 'Phoenix Suns',
  POR: 'Portland Trail Blazers', SAC: 'Sacramento Kings', SAS: 'San Antonio Spurs', TOR: 'Toronto Raptors',
  UTA: 'Utah Jazz', WAS: 'Washington Wizards',
};

// nba team colors (primary)
const TEAM_COLORS: Record<string, string> = {
  ATL: '#E03A3E', BOS: '#007A33', BKN: '#000000', CHA: '#1D1160',
  CHI: '#CE1141', CLE: '#860038', DAL: '#00538C', DEN: '#0E2240',
  DET: '#C8102E', GSW: '#1D428A', HOU: '#CE1141', IND: '#002D62',
  LAC: '#C8102E', LAL: '#552583', MEM: '#5D76A9', MIA: '#98002E',
  MIL: '#00471B', MIN: '#0C2340', NOP: '#0C2340', NYK: '#F58426',
  OKC: '#007AC1', ORL: '#0077C0', PHI: '#006BB6', PHX: '#1D1160',
  POR: '#E03A3E', SAC: '#5A2D81', SAS: '#C4CED4', TOR: '#CE1141',
  UTA: '#002B5C', WAS: '#002B5C',
};

interface TeamsProps {
  season: string;
}

export function Teams({ season }: TeamsProps) {
  const [team1, setTeam1] = useState<string>('');
  const [team2, setTeam2] = useState<string>('');

  const { data: teamsData, isLoading, error } = useTeams(season);
  const { data: quadrantChart, isLoading: quadrantLoading } = useQuadrantChart(season);
  const { data: gridChart, isLoading: gridLoading } = useTeamGridChart(season);
  const { data: radarChart, isLoading: radarLoading } = useTeamRadarChart(team1, team2);

  const teamOptions = useMemo(() => {
    if (!teamsData?.teams) return [];
    return teamsData.teams
      .map((t) => ({ label: TEAM_NAMES[t.Abbrev] || t.Abbrev, value: t.Abbrev }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [teamsData?.teams]);

  useMemo(() => {
    if (teamOptions.length >= 2 && !team1 && !team2) {
      setTeam1(teamOptions[0].value);
      setTeam2(teamOptions[1].value);
    }
  }, [teamOptions, team1, team2]);

  const columns = useMemo(() => [
    {
      key: 'Abbrev',
      header: 'Team',
      render: (_: unknown, row: Team) => (
        <div className="flex items-center gap-2">
          {row.Logo_URL && <img src={row.Logo_URL} alt={row.Abbrev} className="w-5 h-5 object-contain" />}
          <div>
            <div className="text-[#e5e5e5] font-sans">{TEAM_NAMES[row.Abbrev] || row.Abbrev}</div>
            <div className="text-[10px] text-[#666]">{row.Abbrev}</div>
          </div>
        </div>
      ),
    },
    { key: 'WINS', header: 'Record', align: 'center' as const, render: (_: unknown, row: Team) => <span>{row.WINS}-{row.LOSSES}</span> },
    { key: 'Total_Payroll', header: 'Payroll', align: 'right' as const, render: (v: unknown) => <span>${(Number(v) / 1_000_000).toFixed(1)}M</span> },
    {
      key: 'Efficiency_Index',
      header: 'Efficiency',
      align: 'right' as const,
      render: (v: unknown) => {
        const val = Number(v);
        return <span className={val > 0 ? 'text-green' : val < 0 ? 'text-red' : 'text-[#999]'}>{val >= 0 ? '+' : ''}{val.toFixed(2)}</span>;
      },
    },
  ], []);

  const sortedTeams = useMemo(() => {
    if (!teamsData?.teams) return [];
    return [...teamsData.teams].sort((a, b) => (b.Efficiency_Index || 0) - (a.Efficiency_Index || 0));
  }, [teamsData?.teams]);

  if (isLoading) return <PageLoading />;
  if (error) return <ErrorDisplay message={error.message} />;

  return (
    <div className="w-full space-y-6">
      <PageHeader title="Team Efficiency" subtitle="Compare team spending vs performance" />

      <ChartGrid cols={2}>
        <ChartPanel title="Efficiency Quadrant" chartJson={quadrantChart} isLoading={quadrantLoading} height={500} enablePlayerHover={false} />
        <ChartPanel title="Team Grid" chartJson={gridChart} isLoading={gridLoading} height={500} enablePlayerHover={false} />

      </ChartGrid>

      {/* Comparison */}
      <Panel>
        <PanelHeader>Team Comparison</PanelHeader>
        <PanelBody>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-4">
              <Select label="Team 1" value={team1} onChange={setTeam1} options={teamOptions} />
              <Select label="Team 2" value={team2} onChange={setTeam2} options={teamOptions} />
              {team1 && team2 && teamsData?.teams && <ComparisonStats teams={teamsData.teams} team1={team1} team2={team2} />}
            </div>
            <div className="md:col-span-2">
              <ChartPanel chartJson={radarChart} isLoading={radarLoading} height={400} showHeader={false} enablePlayerHover={false} />
            </div>
          </div>
        </PanelBody>
      </Panel>

      <DataTable title="All Teams" subtitle={`${sortedTeams.length} teams ranked by efficiency`} data={sortedTeams} columns={columns} maxHeight="600px" rowKey="Abbrev" />
    </div>
  );
}

interface ComparisonStatsProps {
  teams: Array<{
    Abbrev: string;
    WINS: number;
    LOSSES: number;
    Total_Payroll: number;
    Efficiency_Index: number;
    ConferenceRank?: number;
    PTS?: number;
    OFF_RATING?: number;
    DEF_RATING?: number;
    NET_RATING?: number;
    TS_PCT?: number;
    PACE?: number;
    AST?: number;
    REB?: number;
    Logo_URL?: string;
  }>;
  team1: string;
  team2: string;
}

function ComparisonStats({ teams, team1, team2 }: ComparisonStatsProps) {
  const t1 = teams.find(t => t.Abbrev === team1);
  const t2 = teams.find(t => t.Abbrev === team2);
  if (!t1 || !t2) return null;

  const color1 = TEAM_COLORS[team1] || '#2D96C7';
  const color2 = TEAM_COLORS[team2] || '#2D96C7';

  return (
    <div className="mt-4 space-y-4">
      {/* team logos */}
      <div className="flex items-center justify-between pb-4 border-b border-[#2a2a2a]">
        <div className="flex items-center gap-3">
          {t1.Logo_URL && <img src={t1.Logo_URL} alt={team1} className="w-12 h-12 object-contain" />}
          <div>
            <div className="text-sm font-semibold" style={{ color: color1 }}>{TEAM_NAMES[team1]}</div>
            <div className="text-xs text-[#666]">{team1}</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="text-sm font-semibold" style={{ color: color2 }}>{TEAM_NAMES[team2]}</div>
            <div className="text-xs text-[#666]">{team2}</div>
          </div>
          {t2.Logo_URL && <img src={t2.Logo_URL} alt={team2} className="w-12 h-12 object-contain" />}
        </div>
      </div>

      {/* stats comparison */}
      <div className="space-y-3">
        <ColoredStatCompare label="Wins" value1={t1.WINS} value2={t2.WINS} format={(v) => String(v)} color1={color1} color2={color2} />
        {t1.ConferenceRank && t2.ConferenceRank && (
          <ColoredStatCompare label="Conf Rank" value1={t1.ConferenceRank} value2={t2.ConferenceRank} format={(v) => `#${v}`} color1={color1} color2={color2} inverse />
        )}
        {t1.OFF_RATING && t2.OFF_RATING && (
          <ColoredStatCompare label="Off Rating" value1={t1.OFF_RATING} value2={t2.OFF_RATING} format={(v) => v.toFixed(1)} color1={color1} color2={color2} />
        )}
        {t1.DEF_RATING && t2.DEF_RATING && (
          <ColoredStatCompare label="Def Rating" value1={t1.DEF_RATING} value2={t2.DEF_RATING} format={(v) => v.toFixed(1)} color1={color1} color2={color2} inverse />
        )}
        {t1.NET_RATING && t2.NET_RATING && (
          <ColoredStatCompare label="Net Rating" value1={t1.NET_RATING} value2={t2.NET_RATING} format={(v) => v.toFixed(1)} color1={color1} color2={color2} />
        )}
        {t1.TS_PCT && t2.TS_PCT && (
          <ColoredStatCompare label="TS%" value1={t1.TS_PCT * 100} value2={t2.TS_PCT * 100} format={(v) => `${v.toFixed(1)}%`} color1={color1} color2={color2} />
        )}
        {t1.PACE && t2.PACE && (
          <ColoredStatCompare label="Pace" value1={t1.PACE} value2={t2.PACE} format={(v) => v.toFixed(1)} color1={color1} color2={color2} />
        )}
        <ColoredStatCompare label="Payroll" value1={t1.Total_Payroll} value2={t2.Total_Payroll} format={(v) => `$${(v / 1_000_000).toFixed(0)}M`} color1={color1} color2={color2} inverse />
        <ColoredStatCompare label="Efficiency" value1={t1.Efficiency_Index} value2={t2.Efficiency_Index} format={(v) => v.toFixed(2)} color1={color1} color2={color2} />
      </div>
    </div>
  );
}

interface ColoredStatCompareProps {
  label: string;
  value1: number;
  value2: number;
  format: (v: number) => string;
  color1: string;
  color2: string;
  inverse?: boolean;
}

function ColoredStatCompare({ label, value1, value2, format, color1, color2, inverse = false }: ColoredStatCompareProps) {
  const better1 = inverse ? value1 < value2 : value1 > value2;
  const better2 = inverse ? value2 < value1 : value2 > value1;

  return (
    <div className="flex items-center justify-between text-sm font-mono">
      <span style={{ color: better1 ? color1 : '#666' }} className="font-semibold">{format(value1)}</span>
      <span className="text-[10px] text-[#666] uppercase tracking-wider">{label}</span>
      <span style={{ color: better2 ? color2 : '#666' }} className="font-semibold">{format(value2)}</span>
    </div>
  );
}
