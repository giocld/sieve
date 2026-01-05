/**
 * Lineups Page - Clean trading dashboard style
 */

import { useState, useMemo } from 'react';
import { 
  useLineupTeams, 
  useBestLineups, 
  useWorstLineups,
  useBestLineupsChart,
  useWorstLineupsChart
} from '../hooks/useApi';
import { PageHeader, Panel, PanelHeader, PanelBody, ChartPanel, ChartGrid, DataTable } from '../components';
import { Select, Slider, Button } from '../components/FormControls';
import { ErrorDisplay } from '../components/Loading';

// Component to render player headshots for lineups
function LineupHeadshots({ groupId, groupName }: { groupId?: string; groupName: string }) {
  // GROUP_ID format is typically "playerid1-playerid2" or "playerid1-playerid2-playerid3"
  const playerIds = groupId ? groupId.split('-').filter(id => id && !isNaN(Number(id))) : [];
  const playerNames = groupName.split(' - ');
  
  return (
    <div className="flex items-center gap-2">
      <div className="flex -space-x-2">
        {playerIds.map((id, i) => (
          <img
            key={id}
            src={`https://cdn.nba.com/headshots/nba/latest/260x190/${id}.png`}
            alt=""
            className="w-7 h-7 rounded-full object-cover bg-[#1a1a1a] shrink-0 border-2 border-[#121212]"
            style={{ zIndex: playerIds.length - i }}
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
          />
        ))}
      </div>
      <div className="min-w-0">
        <div className="text-[#e5e5e5] font-sans text-sm truncate">
          {playerNames.map((name, i) => (
            <span key={i}>
              {i > 0 && <span className="text-[#444]"> / </span>}
              {name.trim()}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export function Lineups() {
  const [selectedTeam, setSelectedTeam] = useState<string>('ALL');
  const [lineupSize, setLineupSize] = useState(2);
  const [minMinutes, setMinMinutes] = useState(50);

  const { data: teamsData } = useLineupTeams();

  const teamOptions = useMemo(() => {
    const options = [{ label: 'All Teams', value: 'ALL' }];
    if (teamsData?.teams) options.push(...teamsData.teams);
    return options;
  }, [teamsData?.teams]);

  const filters = useMemo(() => ({
    team: selectedTeam === 'ALL' ? null : selectedTeam,
    size: lineupSize,
    min_minutes: minMinutes,
    limit: 15,
  }), [selectedTeam, lineupSize, minMinutes]);

  const { data: bestLineups, isLoading: bestLoading, error: bestError } = useBestLineups(filters);
  const { data: worstLineups, isLoading: worstLoading, error: worstError } = useWorstLineups(filters);
  const { data: bestChart, isLoading: bestChartLoading } = useBestLineupsChart(filters);
  const { data: worstChart, isLoading: worstChartLoading } = useWorstLineupsChart(filters);

  const isLoading = bestLoading || worstLoading;
  const error = bestError || worstError;

  const columns = useMemo(() => [
    { 
      key: 'GROUP_NAME', 
      header: 'Lineup',
      width: '280px',
      render: (_: unknown, row: Record<string, unknown>) => (
        <LineupHeadshots groupId={row.GROUP_ID as string | undefined} groupName={String(row.GROUP_NAME)} />
      )
    },
    { key: 'MIN', header: 'Minutes', align: 'right' as const, render: (v: unknown) => <span className="text-[#999]">{Number(v).toFixed(1)}</span> },
    {
      key: 'PLUS_MINUS',
      header: '+/-',
      align: 'right' as const,
      render: (v: unknown) => {
        const val = Number(v);
        return <span className={val > 0 ? 'text-green' : val < 0 ? 'text-red' : 'text-[#999]'}>{val > 0 ? '+' : ''}{val.toFixed(1)}</span>;
      },
    },
    {
      key: 'W_PCT',
      header: 'Win %',
      align: 'right' as const,
      render: (v: unknown) => {
        const val = Number(v);
        const pct = (val * 100).toFixed(0);
        return <span className={val >= 0.5 ? 'text-green' : 'text-red'}>{pct}%</span>;
      },
    },
  ], []);

  if (error) return <ErrorDisplay message={error.message} />;

  return (
    <div className="w-full space-y-6">
      <PageHeader title="Lineup Chemistry" subtitle="Discover best and worst performing lineup combinations" />

      {/* Filters */}
      <Panel>
        <PanelBody>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Select label="Team" value={selectedTeam} onChange={setSelectedTeam} options={teamOptions} />
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-[#666] uppercase tracking-wide">Lineup Size</label>
              <div className="flex gap-2">
                <Button variant={lineupSize === 2 ? 'primary' : 'secondary'} size="sm" onClick={() => setLineupSize(2)} className="flex-1">Duos</Button>
                <Button variant={lineupSize === 3 ? 'primary' : 'secondary'} size="sm" onClick={() => setLineupSize(3)} className="flex-1">Trios</Button>
              </div>
            </div>
            <div className="md:col-span-2">
              <Slider label="Minimum Minutes" value={minMinutes} min={10} max={200} step={10} onChange={setMinMinutes} formatValue={(v) => `${v} min`} />
            </div>
          </div>
        </PanelBody>
      </Panel>

      {/* Charts */}
      <ChartGrid cols={2}>
        <ChartPanel title="Best Lineups" chartJson={bestChart} isLoading={bestChartLoading} height={400} />
        <ChartPanel title="Worst Lineups" chartJson={worstChart} isLoading={worstChartLoading} height={400} />
      </ChartGrid>

      {/* Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DataTable title="Best Performing" subtitle={`Top ${bestLineups?.lineups?.length || 0} ${lineupSize === 2 ? 'duos' : 'trios'}`} data={bestLineups?.lineups || []} columns={columns} isLoading={isLoading} maxHeight="400px" />
        <DataTable title="Worst Performing" subtitle={`Bottom ${worstLineups?.lineups?.length || 0} ${lineupSize === 2 ? 'duos' : 'trios'}`} data={worstLineups?.lineups || []} columns={columns} isLoading={isLoading} maxHeight="400px" />
      </div>

      {/* Info */}
      <Panel>
        <PanelHeader>About Lineup Data</PanelHeader>
        <PanelBody>
          <p className="text-sm text-[#999] leading-relaxed">
            Lineup combinations show how well specific player groups perform together. A positive plus/minus indicates the team outscored opponents while this group was on court. The minimum minutes filter ensures statistical significance.
          </p>
        </PanelBody>
      </Panel>
    </div>
  );
}
