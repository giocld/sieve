/**
 * Players Page - Clean trading dashboard style
 */

import { useState, useMemo } from 'react';
import {
  usePlayers,
  useSalaryImpactChart,
  useUnderpaidChart,
  useOverpaidChart,
  useBeeswarmChart
} from '../hooks/useApi';
import {
  PageHeader,
  Panel,
  PanelBody,
  ChartPanel,
  ChartGrid,
  DataTable,
  PlayerCell,
  ValueCell
} from '../components';
import { Slider, Input } from '../components/FormControls';
import { PageLoading, ErrorDisplay } from '../components/Loading';

interface PlayersProps {
  season: string;
}

export function Players({ season }: PlayersProps) {
  const [minLebron, setMinLebron] = useState(-5);
  const [maxLebron, setMaxLebron] = useState(10);
  const [minSalary, setMinSalary] = useState(0);
  const [maxSalary, setMaxSalary] = useState(60);
  const [searchQuery, setSearchQuery] = useState('');

  // all players for search (no filters)
  const { data: allPlayersData } = usePlayers({ season });

  // filtered data for charts and value tables
  const { data: playersData, isLoading, error } = usePlayers({
    season,
    min_lebron: minLebron,
    max_lebron: maxLebron,
    min_salary: minSalary,
    max_salary: maxSalary,
  });


  const { data: scatterChart, isLoading: scatterLoading } = useSalaryImpactChart(season, minLebron, maxLebron, minSalary, maxSalary);
  const { data: underpaidChart, isLoading: underpaidLoading } = useUnderpaidChart(season, minLebron, maxLebron, minSalary, maxSalary);
  const { data: overpaidChart, isLoading: overpaidLoading } = useOverpaidChart(season, minLebron, maxLebron, minSalary, maxSalary);
  const { data: beeswarmChart, isLoading: beeswarmLoading } = useBeeswarmChart(season, minLebron, maxLebron, minSalary, maxSalary);

  // search filters all players by name only, ignoring other filters
  const displayedPlayers = useMemo(() => {
    if (searchQuery.trim()) {
      // when searching, use all players and filter by name only
      if (!allPlayersData?.players) return [];
      const q = searchQuery.trim().toLowerCase();
      return allPlayersData.players.filter((p) => {
        const name = (p.player_name || '').toLowerCase();
        return name.includes(q);
      });
    }
    // when not searching, use filtered players
    return playersData?.players || [];
  }, [playersData?.players, allPlayersData?.players, searchQuery]);

  const { underpaid, overpaid } = useMemo(() => {
    if (!playersData?.players) return { underpaid: [], overpaid: [] };
    const sorted = [...playersData.players].sort((a, b) => (b.value_gap || 0) - (a.value_gap || 0));
    return { underpaid: sorted.slice(0, 10), overpaid: sorted.slice(-10).reverse() };
  }, [playersData?.players]);

  const columns = useMemo(() => [
    {
      key: 'player_name',
      header: 'Player',
      render: (_: unknown, row: typeof displayedPlayers[0]) => (
        <PlayerCell name={row.player_name} playerId={row.PLAYER_ID} team={row.team as string} />
      ),
    },
    {
      key: 'LEBRON',
      header: 'LEBRON',
      align: 'right' as const,
      render: (value: unknown) => (
        <span className={Number(value) >= 0 ? 'text-green' : 'text-red'}>
          {Number(value) >= 0 ? '+' : ''}{Number(value).toFixed(2)}
        </span>
      ),
    },
    { key: 'current_year_salary', header: 'Salary', align: 'right' as const, format: 'currency' as const },
    {
      key: 'value_gap',
      header: 'Value Gap',
      align: 'right' as const,
      render: (value: unknown) => <ValueCell value={Number(value)} />,
    },
  ], []);

  if (isLoading) return <PageLoading />;
  if (error) return <ErrorDisplay message={error.message} />;

  return (
    <div className="w-full space-y-6">
      <PageHeader title="Player Analysis" subtitle={`${playersData?.count || 0} players matching filters`} />

      {/* Filters */}
      <Panel>
        <PanelBody>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Slider label="Min LEBRON" value={minLebron} min={-5} max={maxLebron - 0.5} step={0.5} onChange={setMinLebron} formatValue={(v) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))} />
            <Slider label="Max LEBRON" value={maxLebron} min={minLebron + 0.5} max={10} step={0.5} onChange={setMaxLebron} formatValue={(v) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))} />
            <Slider label="Min Salary" value={minSalary} min={0} max={maxSalary - 1} step={1} onChange={setMinSalary} formatValue={(v) => `$${v}M`} />
            <Slider label="Max Salary" value={maxSalary} min={minSalary + 1} max={60} step={1} onChange={setMaxSalary} formatValue={(v) => `$${v}M`} />
          </div>
        </PanelBody>
      </Panel>

      {/* Charts */}
      <ChartGrid cols={2}>
        <ChartPanel title="Salary vs Impact" chartJson={scatterChart} isLoading={scatterLoading} height={450} />
        <ChartPanel title="Player Distribution" chartJson={beeswarmChart} isLoading={beeswarmLoading} height={450} />
      </ChartGrid>

      <ChartGrid cols={2}>
        <ChartPanel title="Most Underpaid" chartJson={underpaidChart} isLoading={underpaidLoading} height={400} />
        <ChartPanel title="Most Overpaid" chartJson={overpaidChart} isLoading={overpaidLoading} height={400} />
      </ChartGrid>

      {/* Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DataTable title="Top Value Players" data={underpaid} columns={columns} maxHeight="350px" rowKey="PLAYER_ID" />
        <DataTable title="Worst Value Players" data={overpaid} columns={columns} maxHeight="350px" rowKey="PLAYER_ID" />
      </div>

      {/* All Players with search */}
      <DataTable
        title="All Players"
        subtitle={`${displayedPlayers.length} players${searchQuery ? ` matching "${searchQuery}"` : ''}`}
        data={displayedPlayers}
        columns={columns}
        maxHeight="500px"
        rowKey={(row) => row.PLAYER_ID ? String(row.PLAYER_ID) : row.player_name || 'unknown'}
        headerRight={
          <div className="w-64">
            <Input
              placeholder="Search player..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<SearchIcon />}
            />
          </div>
        }
      />
    </div>
  );
}

function SearchIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  );
}
