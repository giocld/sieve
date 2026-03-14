/**
 * DataTable - Clean trading dashboard style with proper sorting
 */

import { useState, useMemo } from 'react';
import { PlayerHoverCard } from './PlayerHoverCard';

interface Column<T> {
  key: keyof T | string;
  header: string;
  width?: string;
  align?: 'left' | 'center' | 'right';
  sortable?: boolean;
  render?: (value: unknown, row: T) => React.ReactNode;
  format?: 'currency' | 'number' | 'percent' | 'none';
  sortKey?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  title?: string;
  subtitle?: string;
  isLoading?: boolean;
  maxHeight?: string;
  className?: string;
  onRowClick?: (row: T) => void;
  rowKey?: keyof T | ((row: T) => string);
  defaultSort?: { key: string; direction: 'asc' | 'desc' };
  headerRight?: React.ReactNode;
}

type SortDirection = 'asc' | 'desc' | null;

function formatValue(value: unknown, format?: string): string {
  if (value === null || value === undefined) return '-';

  switch (format) {
    case 'currency':
      return `$${(Number(value) / 1_000_000).toFixed(1)}M`;
    case 'number':
      return Number(value).toLocaleString();
    case 'percent':
      return `${(Number(value) * 100).toFixed(1)}%`;
    default:
      if (typeof value === 'number') {
        return Number.isInteger(value) ? value.toString() : value.toFixed(2);
      }
      return String(value);
  }
}

function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  return path.split('.').reduce((acc: unknown, part) => {
    if (acc && typeof acc === 'object') {
      return (acc as Record<string, unknown>)[part];
    }
    return undefined;
  }, obj);
}

export function DataTable<T extends Record<string, unknown>>({
  data,
  columns,
  title,
  subtitle,
  isLoading = false,
  maxHeight = '400px',
  className = '',
  onRowClick,
  rowKey,
  defaultSort,
  headerRight,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(defaultSort?.key ?? null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(defaultSort?.direction ?? null);

  const handleSort = (col: Column<T>) => {
    if (col.sortable === false) return;

    const key = col.sortKey || String(col.key);

    if (sortKey === key) {
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortDirection(null);
        setSortKey(null);
      }
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  const sortedData = useMemo(() => {
    if (!sortKey || !sortDirection) return data;

    return [...data].sort((a, b) => {
      const aVal = getNestedValue(a, sortKey);
      const bVal = getNestedValue(b, sortKey);

      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      let cmp = 0;

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        cmp = aVal - bVal;
      } else {
        cmp = String(aVal).localeCompare(String(bVal), undefined, { numeric: true, sensitivity: 'base' });
      }

      return sortDirection === 'desc' ? -cmp : cmp;
    });
  }, [data, sortKey, sortDirection]);

  const getRowKey = (row: T, i: number): string => {
    if (!rowKey) return String(i);
    if (typeof rowKey === 'function') return rowKey(row);
    return String(row[rowKey]);
  };

  const getSortIcon = (col: Column<T>) => {
    if (col.sortable === false) return null;
    const key = col.sortKey || String(col.key);

    if (sortKey !== key) {
      return <span className="text-[#444] ml-1 opacity-0 group-hover:opacity-100 transition-opacity">&#x21C5;</span>;
    }

    return (
      <span className="text-[#3b82f6] ml-1">
        {sortDirection === 'asc' ? '\u2191' : '\u2193'}
      </span>
    );
  };

  return (
    <div className={`panel overflow-hidden ${className}`}>
      {(title || subtitle || headerRight) && (
        <div className="panel-header flex items-center justify-between">
          <div>
            {title && <span>{title}</span>}
            {subtitle && <span className="ml-2 text-[#666] font-normal">{subtitle}</span>}
          </div>
          {headerRight}
        </div>
      )}

      <div className="overflow-auto" style={{ maxHeight }}>
        <table className="data-table">
          <thead className="sticky top-0 z-10">
            <tr>
              {columns.map((col) => {
                const isSortable = col.sortable !== false;
                return (
                  <th
                    key={String(col.key)}
                    className={`
                      ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'}
                      ${isSortable ? 'cursor-pointer select-none group hover:text-[#999] transition-colors' : ''}
                    `}
                    style={{ width: col.width }}
                    onClick={() => handleSort(col)}
                  >
                    <div className={`flex items-center gap-0.5 ${col.align === 'right' ? 'justify-end' : col.align === 'center' ? 'justify-center' : ''}`}>
                      <span>{col.header}</span>
                      {getSortIcon(col)}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}>
                  {columns.map((_, j) => (
                    <td key={j}><div className="h-4 skeleton rounded" /></td>
                  ))}
                </tr>
              ))
            ) : sortedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="py-8 text-center text-[#666] text-sm font-sans">
                  No data available
                </td>
              </tr>
            ) : (
              sortedData.map((row, i) => (
                <tr
                  key={getRowKey(row, i)}
                  className={onRowClick ? 'cursor-pointer' : ''}
                  onClick={() => onRowClick?.(row)}
                >
                  {columns.map((col) => {
                    const value = getNestedValue(row, String(col.key));
                    return (
                      <td
                        key={String(col.key)}
                        className={col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'}
                      >
                        {col.render ? col.render(value, row) : formatValue(value, col.format)}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * PlayerCell - with headshot and hover card
 */
interface PlayerCellProps {
  name: string;
  playerId?: number;
  team?: string;
  playerData?: any;
}

export function PlayerCell({ name, playerId, team, playerData }: PlayerCellProps) {
  // Construct data for hover card if not provided fully
  const data = playerData || {
    player_name: name,
    player_id: playerId,
    team: team
  };

  return (
    <PlayerHoverCard player={data}>
      <div className="flex items-center gap-2 group cursor-default">
        {playerId && (
          <img
            src={`https://cdn.nba.com/headshots/nba/latest/260x190/${playerId}.png`}
            alt=""
            className="w-8 h-8 rounded-full object-cover bg-[#1a1a1a] shrink-0"
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
          />
        )}
        <div className="min-w-0">
          <div className="text-[#e5e5e5] font-sans truncate group-hover:text-[#3b82f6] transition-colors">{name}</div>
          {team && <div className="text-[10px] text-[#666]">{team}</div>}
        </div>
      </div>
    </PlayerHoverCard>
  );
}

/**
 * ValueCell
 */
interface ValueCellProps {
  value: number;
  format?: 'currency' | 'number' | 'percent';
}

export function ValueCell({ value, format }: ValueCellProps) {
  const color = value > 0 ? 'text-green' : value < 0 ? 'text-red' : 'text-[#999]';
  return <span className={color}>{value > 0 ? '+' : ''}{formatValue(value, format)}</span>;
}

/**
 * PlayerHeadshot - standalone headshot component
 */
interface PlayerHeadshotProps {
  playerId?: number;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function PlayerHeadshot({ playerId, size = 'md', className = '' }: PlayerHeadshotProps) {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-10 h-10',
  };

  if (!playerId) return null;

  return (
    <img
      src={`https://cdn.nba.com/headshots/nba/latest/260x190/${playerId}.png`}
      alt=""
      className={`${sizeClasses[size]} rounded-full object-cover bg-[#1a1a1a] shrink-0 ${className}`}
      onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
    />
  );
}
