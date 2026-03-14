/**
 * Similarity Page - Clean trading dashboard style
 */

import { useState, useMemo } from 'react';
import {
  useSimilarityPlayers,
  usePlayerSeasons,
  useSimilarPlayers,
  useDiamondFinderPlayers,
  useDiamondReplacements
} from '../hooks/useApi';
import { PageHeader, Panel, PanelHeader, PanelBody, MetricCard, PlayerHoverCard } from '../components';
import { Select, Button } from '../components/FormControls';
import { PageLoading, ErrorDisplay } from '../components/Loading';

interface SimilarityProps {
  season: string;
}

export function Similarity({ season }: SimilarityProps) {
  const [activeTab, setActiveTab] = useState<'similarity' | 'diamond'>('similarity');

  return (
    <div className="w-full space-y-6">
      <PageHeader title="Similarity Engine" subtitle="Find historical player comparisons and budget-friendly replacements" />

      {/* Tabs */}
      <div className="flex gap-2 border-b border-[#2a2a2a] pb-4">
        <Button variant={activeTab === 'similarity' ? 'primary' : 'ghost'} onClick={() => setActiveTab('similarity')}>Historical Comparisons</Button>
        <Button variant={activeTab === 'diamond' ? 'primary' : 'ghost'} onClick={() => setActiveTab('diamond')}>Diamond Finder</Button>
      </div>

      {activeTab === 'similarity' ? <SimilarityTab /> : <DiamondFinderTab season={season} />}
    </div>
  );
}

function SimilarityTab() {
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [selectedSeason, setSelectedSeason] = useState('');
  const [excludeSelf, setExcludeSelf] = useState(true);

  const { data: playersData, isLoading: playersLoading } = useSimilarityPlayers();
  const { data: seasonsData } = usePlayerSeasons(selectedPlayer);
  const { data: similarData, isLoading: similarLoading, error } = useSimilarPlayers(selectedPlayer, selectedSeason, excludeSelf);

  const playerOptions = useMemo(() => {
    if (!playersData?.players) return [];
    return playersData.players.map(p => ({ label: p, value: p }));
  }, [playersData?.players]);

  const seasonOptions = useMemo(() => {
    if (!seasonsData?.seasons) return [];
    return seasonsData.seasons.map(s => ({ label: s, value: s }));
  }, [seasonsData?.seasons]);

  useMemo(() => {
    if (seasonOptions.length > 0 && !selectedSeason) setSelectedSeason(seasonOptions[0].value);
  }, [seasonOptions, selectedSeason]);

  const handlePlayerChange = (value: string) => {
    setSelectedPlayer(value);
    setSelectedSeason('');
  };

  if (playersLoading) return <PageLoading />;

  return (
    <div className="w-full space-y-6">
      {/* Controls */}
      <Panel>
        <PanelBody>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Select label="Select Player" value={selectedPlayer} onChange={handlePlayerChange} options={[{ label: 'Choose a player...', value: '' }, ...playerOptions]} />
            <Select label="Season" value={selectedSeason} onChange={setSelectedSeason} options={seasonOptions.length > 0 ? seasonOptions : [{ label: 'Select player first', value: '' }]} disabled={!selectedPlayer} />
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-[#666] uppercase tracking-wide">Options</label>
              <Button variant={excludeSelf ? 'primary' : 'secondary'} size="sm" onClick={() => setExcludeSelf(!excludeSelf)}>
                {excludeSelf ? 'Excluding Same Player' : 'Including Same Player'}
              </Button>
            </div>
          </div>
        </PanelBody>
      </Panel>

      {error && <ErrorDisplay message={error.message} />}

      {similarLoading && selectedPlayer && selectedSeason && (
        <div className="flex justify-center py-8">
          <div className="w-6 h-6 border-2 border-[#3b82f6] border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {similarData && similarData.target && (
        <div className="space-y-4">
          {/* Target */}
          <Panel>
            <PanelBody>
              <div className="flex items-center gap-4">
                {similarData.target.player_id ? (
                  <img
                    src={`https://cdn.nba.com/headshots/nba/latest/260x190/${similarData.target.player_id}.png`}
                    alt=""
                    className="w-12 h-12 rounded-full object-cover bg-[#1a1a1a] shrink-0 ring-2 ring-[#3b82f6]"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                ) : (
                  <div className="w-12 h-12 rounded-full bg-[#3b82f6]/20 flex items-center justify-center">
                    <span className="text-[#3b82f6] font-mono font-bold text-sm">TGT</span>
                  </div>
                )}
                <div>
                  <h3 className="text-lg font-semibold text-[#3b82f6]">{similarData.target.name}</h3>
                  <p className="text-sm text-[#666]">{similarData.target.season} - {similarData.target.position}</p>
                </div>
              </div>
            </PanelBody>
          </Panel>

          <div className="section-label">Most Similar Players</div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {similarData.similar.map((player, i) => (
              <SimilarPlayerCard key={`${player.Player}-${player.Season}`} rank={i + 1} player={player} />
            ))}
          </div>
        </div>
      )}

      {!selectedPlayer && (
        <Panel>
          <PanelBody className="text-center py-12">
            <p className="text-[#666]">Select a player and season to find similar historical performances</p>
          </PanelBody>
        </Panel>
      )}
    </div>
  );
}

interface SimilarPlayerCardProps {
  rank: number;
  player: { Player: string; Season: string; id: number; Stats: Record<string, number>; MatchScore?: number };
}

function SimilarPlayerCard({ rank, player }: SimilarPlayerCardProps) {
  const hoverData = {
    player_name: player.Player,
    player_id: player.id,
    lebron: player.Stats?.LEBRON, // Assuming LEBRON might be in stats
  };

  return (
    <Panel>
      <PanelBody>
        <PlayerHoverCard player={hoverData}>
          <div className="flex items-start gap-3 mb-4 cursor-default">
            <div className="relative">
              {player.id ? (
                <img
                  src={`https://cdn.nba.com/headshots/nba/latest/260x190/${player.id}.png`}
                  alt=""
                  className="w-10 h-10 rounded-full object-cover bg-[#1a1a1a] shrink-0"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
              ) : null}
              <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-[#1a1a1a] flex items-center justify-center text-[10px] font-mono text-[#666] border border-[#2a2a2a]">{rank}</div>
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="font-medium text-[#e5e5e5] group-hover:text-[#3b82f6] transition-colors">{player.Player}</h4>
              <p className="text-xs text-[#666]">{player.Season}</p>
              {player.MatchScore && <p className="text-xs text-blue mt-1">{player.MatchScore.toFixed(0)}% match</p>}
            </div>
          </div>
        </PlayerHoverCard>
        <div className="grid grid-cols-3 gap-2 pt-3 border-t border-[#2a2a2a]">
          {Object.entries(player.Stats).slice(0, 3).map(([key, value]) => (
            <div key={key} className="text-center">
              <div className="text-sm font-mono text-[#e5e5e5]">{typeof value === 'number' ? value.toFixed(1) : value}</div>
              <div className="text-[10px] text-[#666] uppercase">{key}</div>
            </div>
          ))}
        </div>
      </PanelBody>
    </Panel>
  );
}

interface DiamondFinderTabProps {
  season: string;
}

function DiamondFinderTab({ season }: DiamondFinderTabProps) {
  const [selectedPlayer, setSelectedPlayer] = useState('');

  const { data: playersData, isLoading: playersLoading } = useDiamondFinderPlayers(season);
  const { data: replacementData, isLoading: replacementLoading, error } = useDiamondReplacements(selectedPlayer, season);

  const playerOptions = useMemo(() => {
    if (!playersData?.players) return [];
    return playersData.players.map(p => ({
      label: `${p.name} - $${(p.salary / 1_000_000).toFixed(1)}M (LEBRON: ${p.lebron.toFixed(2)})`,
      value: p.name,
    }));
  }, [playersData?.players]);

  if (playersLoading) return <PageLoading />;

  return (
    <div className="w-full space-y-6">
      <Panel>
        <PanelHeader>Find Cheaper Replacements</PanelHeader>
        <PanelBody>
          <p className="text-sm text-[#999] mb-4">Select a high-salary player to find similar performers at a lower cost.</p>
          <Select label="Select Player to Replace" value={selectedPlayer} onChange={setSelectedPlayer} options={[{ label: 'Choose a player...', value: '' }, ...playerOptions]} />
        </PanelBody>
      </Panel>

      {error && <ErrorDisplay message={error.message} />}

      {replacementLoading && selectedPlayer && (
        <div className="flex justify-center py-8">
          <div className="w-6 h-6 border-2 border-[#3b82f6] border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {replacementData && (
        <div className="space-y-4">
          {/* Target */}
          <Panel>
            <PanelBody>
              <div className="flex items-center gap-4">
                {replacementData.target.player_id ? (
                  <img
                    src={`https://cdn.nba.com/headshots/nba/latest/260x190/${replacementData.target.player_id}.png`}
                    alt=""
                    className="w-12 h-12 rounded-full object-cover bg-[#1a1a1a] shrink-0 ring-2 ring-[#ef4444]"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                ) : (
                  <div className="w-12 h-12 rounded-full bg-[#ef4444]/20 flex items-center justify-center">
                    <span className="text-[#ef4444] font-mono font-bold text-sm">RPL</span>
                  </div>
                )}
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-[#ef4444]">{replacementData.target.name}</h3>
                  <div className="flex items-center gap-4 text-sm text-[#666] mt-1">
                    <span>${(replacementData.target.salary / 1_000_000).toFixed(1)}M</span>
                    <span>LEBRON: {replacementData.target.lebron.toFixed(2)}</span>
                  </div>
                  <div className="flex gap-2 mt-2">
                    <span className="badge badge-blue">{replacementData.target.archetype}</span>
                    <span className="badge" style={{ background: 'rgba(153,153,153,0.15)', color: '#999' }}>{replacementData.target.defense_role}</span>
                  </div>
                </div>
              </div>
            </PanelBody>
          </Panel>

          <div className="section-label">Potential Replacements ({replacementData.replacements.length})</div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {replacementData.replacements.map((r, i) => (
              <ReplacementCard key={r.player_name} rank={i + 1} replacement={r} targetSalary={replacementData.target.salary} />
            ))}
          </div>

          {replacementData.replacements.length === 0 && (
            <Panel>
              <PanelBody className="text-center py-8">
                <p className="text-[#666]">No suitable replacements found.</p>
              </PanelBody>
            </Panel>
          )}
        </div>
      )}

      {!selectedPlayer && (
        <Panel>
          <PanelBody className="text-center py-12">
            <p className="text-[#666]">Select a player above to find cheaper alternatives</p>
          </PanelBody>
        </Panel>
      )}
    </div>
  );
}

interface ReplacementCardProps {
  rank: number;
  replacement: { player_name: string; PLAYER_ID?: number; salary: number; LEBRON: number; match_score: number; archetype?: string; defense_role?: string };
  targetSalary: number;
}

function ReplacementCard({ rank, replacement, targetSalary }: ReplacementCardProps) {
  const savings = targetSalary - replacement.salary;
  const savingsPercent = (savings / targetSalary) * 100;
  const lebron = replacement.LEBRON ?? 0;

  const hoverData = {
    player_name: replacement.player_name,
    player_id: replacement.PLAYER_ID,
    salary: replacement.salary,
    lebron: replacement.LEBRON,
    archetype: replacement.archetype,
    role: replacement.defense_role
  };

  return (
    <Panel>
      <PanelBody>
        <PlayerHoverCard player={hoverData}>
          <div className="flex items-center gap-4 mb-4 cursor-default">
            <div className="relative">
              {replacement.PLAYER_ID ? (
                <img
                  src={`https://cdn.nba.com/headshots/nba/latest/260x190/${replacement.PLAYER_ID}.png`}
                  alt=""
                  className="w-10 h-10 rounded-full object-cover bg-[#1a1a1a] shrink-0"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
              ) : (
                <div className="w-10 h-10 rounded-full bg-[#3b82f6]/20 flex items-center justify-center">
                  <span className="text-blue font-mono text-sm">{rank}</span>
                </div>
              )}
              {replacement.PLAYER_ID && (
                <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-[#3b82f6] flex items-center justify-center text-[10px] font-mono text-white font-bold">{rank}</div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="font-medium text-[#e5e5e5] group-hover:text-[#3b82f6] transition-colors">{replacement.player_name}</h4>
              <p className="text-xs text-blue">{replacement.match_score?.toFixed(0)}% similar</p>
              <div className="flex gap-1 mt-1 flex-wrap">
                {replacement.archetype && <span className="badge badge-blue" style={{ fontSize: '9px', padding: '1px 6px' }}>{replacement.archetype}</span>}
                {replacement.defense_role && <span className="badge" style={{ background: 'rgba(153,153,153,0.15)', color: '#999', fontSize: '9px', padding: '1px 6px' }}>{replacement.defense_role}</span>}
              </div>
            </div>
          </div>
        </PlayerHoverCard>
        <div className="grid grid-cols-3 gap-3">
          <MetricCard label="Salary" value={`$${(replacement.salary / 1_000_000).toFixed(1)}M`} color="blue" size="sm" />
          <MetricCard label="LEBRON" value={lebron >= 0 ? `+${lebron.toFixed(2)}` : lebron.toFixed(2)} color={lebron >= 0 ? 'green' : 'red'} size="sm" />
          <MetricCard label="Savings" value={`$${(savings / 1_000_000).toFixed(1)}M`} subValue={`${savingsPercent.toFixed(0)}%`} color="green" size="sm" />
        </div>
      </PanelBody>
    </Panel>
  );
}
