
import { useState } from 'react';
import { createPortal } from 'react-dom';

interface PlayerData {
    player_name?: string;
    player_id?: number;
    team?: string;
    lebron?: number;
    o_lebron?: number;
    d_lebron?: number;
    salary?: number;
    value_gap?: number;
    archetype?: string;
    role?: string;
    ppg?: number;
    rpg?: number;
    apg?: number;
    spg?: number;
    bpg?: number;
    fg_pct?: number;
    three_pct?: number;
    ft_pct?: number;
    ts_pct?: number;
    ppg_pct?: number;
    rpg_pct?: number;
    apg_pct?: number;
    spg_pct?: number;
    bpg_pct?: number;
    fg_pct_pct?: number;
    three_pct_pct?: number;
    ft_pct_pct?: number;
    ts_pct_pct?: number;
}

interface PlayerHoverCardProps {
    player: PlayerData;
    children?: React.ReactNode;
    manualPosition?: { x: number; y: number } | null;
    className?: string; // wrapper class
}

export function PlayerHoverCard({ player, children, manualPosition, className = '' }: PlayerHoverCardProps) {
    const [isHovered, setIsHovered] = useState(false);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

    // Use manual position if provided (chart mode), otherwise tracking state (table mode)
    const show = manualPosition ? true : isHovered;
    const position = manualPosition || mousePos;

    // Handle auto-closing if no manual position and mouse leaves
    const handleMouseEnter = (e: React.MouseEvent) => {
        if (manualPosition) return;
        setIsHovered(true);
        updatePosition(e);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (manualPosition) return;
        updatePosition(e);
    };

    const handleMouseLeave = () => {
        if (manualPosition) return;
        setIsHovered(false);
    };

    const updatePosition = (e: React.MouseEvent) => {
        setMousePos({ x: e.clientX, y: e.clientY });
    };

    // Content of the card
    const cardContent = (
        <div
            className="fixed z-[9999] pointer-events-none animate-fade-in"
            style={{
                left: position.x,
                top: position.y - 12, // slightly above cursor
                transform: 'translate(-50%, -100%)', // Center horizontally, place above
            }}
        >
            <div className="bg-[#1a1a1a]/95 backdrop-blur-sm border border-[#333] rounded-xl shadow-2xl overflow-hidden min-w-[320px]">
                {/* Header */}
                <div className="flex items-center gap-3 p-3 border-b border-[#2a2a2a] bg-[#141414]/50">
                    {player.player_id && (
                        <img
                            src={`https://cdn.nba.com/headshots/nba/latest/260x190/${player.player_id}.png`}
                            alt=""
                            className="w-12 h-12 rounded-full object-cover bg-[#0a0a0a] border border-[#333]"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                    )}
                    <div className="flex-1 min-w-0">
                        <div className="text-white font-bold text-base truncate">{player.player_name || 'Unknown Player'}</div>
                        <div className="flex items-center gap-2 text-[11px] mt-0.5">
                            {player.team && <span className="font-mono text-[#3b82f6] font-medium">{player.team}</span>}
                            {player.role && <span className="text-[#888] truncate border-l border-[#333] pl-2">{player.role}</span>}
                        </div>
                    </div>
                    {player.salary !== undefined && player.salary > 0 && (
                        <div className="text-right shrink-0 bg-[#0f0f0f] px-2 py-1 rounded border border-[#222]">
                            <div className="text-[#22c55e] font-mono font-bold text-sm">
                                ${(player.salary / 1_000_000).toFixed(1)}M
                            </div>
                        </div>
                    )}
                </div>

                {/* Main Stats Grid */}
                <div className="p-3 border-b border-[#2a2a2a]">
                    <div className="grid grid-cols-5 gap-2 text-center">
                        <StatBox label="PPG" value={player.ppg} pct={player.ppg_pct} />
                        <StatBox label="RPG" value={player.rpg} pct={player.rpg_pct} />
                        <StatBox label="APG" value={player.apg} pct={player.apg_pct} />
                        <StatBox label="SPG" value={player.spg} pct={player.spg_pct} />
                        <StatBox label="BPG" value={player.bpg} pct={player.bpg_pct} />
                    </div>
                </div>

                {/* Shooting Splits */}
                <div className="px-3 py-2 border-b border-[#2a2a2a] bg-[#141414]/30">
                    <div className="text-[10px] text-[#666] mb-1 font-semibold uppercase tracking-wider">Shooting Efficiency</div>
                    <div className="grid grid-cols-4 gap-2 text-center">
                        <StatBox label="FG%" value={player.fg_pct && player.fg_pct * 100} pct={player.fg_pct_pct} suffix="%" fixed={1} />
                        <StatBox label="3P%" value={player.three_pct && player.three_pct * 100} pct={player.three_pct_pct} suffix="%" fixed={1} />
                        <StatBox label="FT%" value={player.ft_pct && player.ft_pct * 100} pct={player.ft_pct_pct} suffix="%" fixed={1} />
                        <StatBox label="TS%" value={player.ts_pct && player.ts_pct * 100} pct={player.ts_pct_pct} suffix="%" fixed={1} highlight />
                    </div>
                </div>

                {/* Advanced Metrics */}
                <div className="grid grid-cols-2 divide-x divide-[#2a2a2a]">
                    <div className="p-3 flex items-center justify-between">
                        <span className="text-xs text-[#888] font-medium">LEBRON</span>
                        <div className="text-right">
                            <span className={`text-lg font-bold ${getScoreColor(player.lebron)}`}>
                                {formatVal(player.lebron)}
                            </span>
                        </div>
                    </div>
                    <div className="p-3 flex items-center justify-between">
                        <span className="text-xs text-[#888] font-medium">Value Gap</span>
                        <div className="text-right">
                            <span className={`text-lg font-bold ${getValColor(player.value_gap)}`}>
                                {player.value_gap && player.value_gap > 0 ? '+' : ''}{formatVal(player.value_gap, 1)}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Archetype Footer */}
                {player.archetype && (
                    <div className="px-3 py-2 bg-[#0f0f0f] border-t border-[#2a2a2a] text-center">
                        <span className="text-[10px] text-[#888] tracking-tight">{player.archetype}</span>
                    </div>
                )}
            </div>
        </div>
    );

    return (
        <>
            <div
                className={className}
                onMouseEnter={handleMouseEnter}
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
            >
                {children}
            </div>
            {show && createPortal(cardContent, document.body)}
        </>
    );
}

// Helper components and functions

function StatBox({ label, value, pct, suffix = '', fixed = 1, highlight = false }: { label: string; value?: number; pct?: number; suffix?: string; fixed?: number; highlight?: boolean }) {
    if (value === undefined || value === null) return <div className="flex flex-col items-center opacity-30"><span className="text-[9px] uppercase">{label}</span><span className="text-sm">-</span></div>;

    // Percentile color
    const getPctColor = (p?: number) => {
        if (!p) return 'bg-[#333]';
        if (p >= 0.9) return 'bg-[#06d6a0]'; // Top 10%
        if (p >= 0.75) return 'bg-[#22c55e]'; // Top 25%
        if (p >= 0.5) return 'bg-[#eab308]';  // Above Avg
        if (p >= 0.25) return 'bg-[#f59e0b]'; // Below Avg
        return 'bg-[#ef4444]';                // Bottom 25%
    };

    return (
        <div className="flex flex-col items-center">
            <span className={`text-[9px] uppercase tracking-wider mb-0.5 ${highlight ? 'text-[#3b82f6] font-bold' : 'text-[#888]'}`}>{label}</span>
            <span className={`font-mono font-bold leading-none ${highlight ? 'text-[#3b82f6]' : 'text-[#e5e5e5]'} text-sm`}>
                {typeof value === 'number' ? value.toFixed(fixed) : value}{suffix}
            </span>

            {/* Percentile Bar */}
            {pct !== undefined && typeof pct === 'number' && (
                <div className="w-full max-w-[40px] h-1 bg-[#2a2a2a] rounded-full mt-1.5 overflow-hidden">
                    <div
                        className={`h-full rounded-full ${getPctColor(pct)}`}
                        style={{ width: `${Math.min(pct * 100, 100)}%` }}
                    />
                </div>
            )}

            {pct !== undefined && typeof pct === 'number' && pct >= 0.9 && (
                <span className="text-[8px] text-[#06d6a0] mt-0.5">Top {Math.max(1, Math.round((1 - pct) * 100))}%</span>
            )}
        </div>
    );
}

const getScoreColor = (val?: number) => {
    if (val === undefined) return 'text-[#888]';
    if (val >= 2.0) return 'text-[#06d6a0]';
    if (val >= 0) return 'text-[#ffd166]';
    return 'text-[#ef4444]';
};

const getValColor = (val?: number) => {
    if (val === undefined) return 'text-[#888]';
    if (val > 5.0) return 'text-[#06d6a0]';
    if (val > 0) return 'text-[#ffd166]';
    return 'text-[#ef4444]';
};

const formatVal = (val?: number, fixed = 2) => {
    if (val === undefined) return '-';
    return val.toFixed(fixed);
};
