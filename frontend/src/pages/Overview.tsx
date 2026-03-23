/**
 * Overview Page - Clean trading dashboard style
 */

import { Link } from 'react-router-dom';
import { useOverview, useTeams } from '../hooks/useApi';
import { PageHeader, PlayerCell } from '../components';
import { PageLoading, ErrorDisplay } from '../components/Loading';
import type { OverviewPlayer } from '../lib/api';

interface OverviewProps {
  season: string;
}

export function Overview({ season }: OverviewProps) {
  const { data: overview, isLoading, error } = useOverview(season);
  const { data: teamsData } = useTeams(season);

  if (isLoading) return <PageLoading />;
  if (error) return <ErrorDisplay message={error.message} />;
  if (!overview) return <ErrorDisplay message="No data available" />;

  const topTeams = teamsData?.teams
    .sort((a, b) => b.Efficiency_Index - a.Efficiency_Index)
    .slice(0, 5) || [];

  return (
    <div className="w-full space-y-6">
      <PageHeader
        title="Overview"
        subtitle="NBA Player Value & Efficiency Analysis Platform"
      />

      {/* Top Stats Bar */}
      <div className="flex flex-wrap items-center gap-x-8 gap-y-2 py-3 border-b border-[#2a2a2a]">
        <StatItem label="Players" value={overview.num_players} />
        <StatItem label="Teams" value={overview.num_teams} />
        <StatItem label="Avg Salary" value={`$${overview.avg_salary_millions.toFixed(1)}M`} color="text-blue" />
        <StatItem
          label="Avg LEBRON"
          value={overview.avg_lebron >= 0 ? `+${overview.avg_lebron.toFixed(2)}` : overview.avg_lebron.toFixed(2)}
          color={overview.avg_lebron >= 0 ? 'text-green' : 'text-red'}
        />
        <StatItem label="League Payroll" value={`$${overview.total_payroll_billions.toFixed(1)}B`} color="text-blue" />
        <StatItem label="Most Efficient" value={overview.most_efficient_team || 'N/A'} color="text-green" />
      </div>

      {/* Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <NavCard to="/players" title="Player Analysis" desc="Explore player value, contracts, and performance metrics" />
        <NavCard to="/teams" title="Team Efficiency" desc="Compare team spending vs performance across the league" />
        <NavCard to="/similarity" title="Similarity Engine" desc="Find historical player comparisons and replacements" />
      </div>

      {/* Player Lists - symmetric 3 column */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <PlayerList
          title="Best Value Players"
          badge="Underpaid"
          badgeClass="badge-green"
          players={overview.top_value_players || []}
          type="value"
        />
        <PlayerList
          title="Top Performers"
          badge="LEBRON"
          badgeClass="badge-blue"
          players={overview.top_performers || []}
          type="lebron"
        />
        <PlayerList
          title="Worst Value Players"
          badge="Overpaid"
          badgeClass="badge-red"
          players={overview.worst_value_players || []}
          type="overpaid"
        />
      </div>

      {/* Bottom Row - symmetric 2 column */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Team Leaderboard */}
        <div className="panel">
          <div className="panel-header">Team Efficiency Leaders</div>
          <div className="panel-body p-0">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-10">#</th>
                  <th>Team</th>
                  <th className="text-center">Record</th>
                  <th className="text-right">Index</th>
                </tr>
              </thead>
              <tbody>
                {topTeams.map((team, i) => (
                  <tr key={team.Abbrev}>
                    <td className="text-[#666]">{i + 1}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        {team.Logo_URL && <img src={team.Logo_URL} alt="" className="w-5 h-5 object-contain" />}
                        <span className="font-sans text-[#e5e5e5]">{team.Abbrev}</span>
                      </div>
                    </td>
                    <td className="text-center text-[#999]">{team.WINS}W-{team.LOSSES}L</td>
                    <td className="text-right text-green">{team.Efficiency_Index?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="px-4 py-3 border-t border-[#2a2a2a]">
              <Link to="/teams" className="text-xs text-[#666] hover:text-[#3b82f6]">View all teams</Link>
            </div>
          </div>
        </div>

        {/* Metrics Explained */}
        <div className="panel">
          <div className="panel-header">Key Metrics</div>
          <div className="panel-body space-y-4">
            <MetricInfo
              name="LEBRON"
              range="-3 to +6"
              desc="Luck-adjusted Estimate of Box-score and Real On-court Network. Measures overall player impact per game."
            />
            <MetricInfo
              name="Value Gap"
              range="-100 to +100"
              desc="Difference between normalized impact and salary. Positive = underpaid, negative = overpaid."
            />
            <MetricInfo
              name="Efficiency Index"
              range="0 to 5+"
              desc="Team wins relative to payroll spending. Higher values = better value for money."
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// Stat item for top bar
function StatItem({ label, value, color = 'text-[#e5e5e5]' }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="stat-inline">
      <span className="label">{label}</span>
      <span className={`value ${color}`}>{value}</span>
    </div>
  );
}

// Nav card
function NavCard({ to, title, desc }: { to: string; title: string; desc: string }) {
  return (
    <Link to={to} className="panel p-4 hover:border-[#333] transition-colors">
      <h3 className="font-medium text-[#e5e5e5] mb-1">{title}</h3>
      <p className="text-xs text-[#666] leading-relaxed">{desc}</p>
    </Link>
  );
}

// Player list panel
function playersDeepLink(type: 'value' | 'lebron' | 'overpaid'): string {
  if (type === 'value') return '/players#top-value';
  if (type === 'overpaid') return '/players#worst-value';
  return '/players#all-players';
}

function PlayerList({
  title,
  badge,
  badgeClass,
  players,
  type
}: {
  title: string;
  badge: string;
  badgeClass: string;
  players: OverviewPlayer[];
  type: 'value' | 'lebron' | 'overpaid';
}) {
  return (
    <div className="panel flex flex-col">
      <div className="panel-header flex items-center justify-between">
        <span>{title}</span>
        <span className={`badge ${badgeClass}`}>{badge}</span>
      </div>
      <div className="panel-body p-0 flex-1">
        <table className="data-table">
          <thead>
            <tr>
              <th className="w-8">#</th>
              <th>Player</th>
              <th className="text-right">{type === 'lebron' ? 'Score' : 'Value'}</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p, i) => {
              const val = type === 'lebron'
                ? `+${p.lebron.toFixed(2)}`
                : (p.value_gap >= 0 ? '+' : '') + p.value_gap.toFixed(1);
              const valColor = type === 'overpaid' ? 'text-red' : type === 'lebron' ? 'text-blue' : 'text-green';

              return (
                <tr key={p.name}>
                  <td className="text-[#666]">{i + 1}</td>
                  <td>
                    <div className="flex items-center gap-2">
                      <PlayerCell
                        name={p.name}
                        playerId={p.player_id}
                        team={`${p.team} - $${p.salary.toFixed(1)}M`}
                        playerData={{
                          player_name: p.name,
                          player_id: p.player_id,
                          team: p.team,
                          salary: p.salary * 1_000_000,
                          lebron: p.lebron,
                          value_gap: p.value_gap,
                          o_lebron: p.o_lebron,
                          d_lebron: p.d_lebron,
                          role: p.role,
                          archetype: p.archetype,
                          ppg: p.ppg,
                          rpg: p.rpg,
                          apg: p.apg,
                          spg: p.spg,
                          bpg: p.bpg,
                          fg_pct: p.fg_pct,
                          three_pct: p.three_pct,
                          ft_pct: p.ft_pct,
                          ts_pct: p.ts_pct,
                          ppg_pct: p.ppg_pct,
                          rpg_pct: p.rpg_pct,
                          apg_pct: p.apg_pct,
                          spg_pct: p.spg_pct,
                          bpg_pct: p.bpg_pct,
                          fg_pct_pct: p.fg_pct_pct,
                          three_pct_pct: p.three_pct_pct,
                          ft_pct_pct: p.ft_pct_pct,
                          ts_pct_pct: p.ts_pct_pct,
                        }}
                      />
                    </div>
                  </td>
                  <td className={`text-right ${valColor}`}>{val}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="px-4 py-3 border-t border-[#2a2a2a] mt-auto">
          <Link to={playersDeepLink(type)} className="text-xs text-[#666] hover:text-[#3b82f6]">
            {type === 'value' && 'View top value players'}
            {type === 'overpaid' && 'View worst value players'}
            {type === 'lebron' && 'View all players'}
          </Link>
        </div>
      </div>
    </div>
  );
}

// Metric info
function MetricInfo({ name, range, desc }: { name: string; range: string; desc: string }) {
  return (
    <div className="border-b border-[#2a2a2a] pb-4 last:border-b-0 last:pb-0">
      <div className="flex items-center justify-between mb-1">
        <span className="font-medium text-[#3b82f6]">{name}</span>
        <span className="font-mono text-xs text-[#666]">{range}</span>
      </div>
      <p className="text-xs text-[#999] leading-relaxed">{desc}</p>
    </div>
  );
}
