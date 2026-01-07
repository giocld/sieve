/**
 * MetricCard - Clean trading dashboard style
 */

import type { ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  change?: number;
  icon?: ReactNode;
  color?: 'green' | 'red' | 'blue' | 'yellow' | 'default';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const colorClasses = {
  green: 'text-green',
  red: 'text-red',
  blue: 'text-blue',
  yellow: 'text-yellow',
  default: 'text-[#e5e5e5]',
};

const sizeClasses = {
  sm: { value: 'text-base', label: 'text-[10px]', pad: 'p-3' },
  md: { value: 'text-lg', label: 'text-[11px]', pad: 'p-4' },
  lg: { value: 'text-xl', label: 'text-xs', pad: 'p-5' },
};

export function MetricCard({
  label,
  value,
  subValue,
  change,
  icon,
  color = 'default',
  size = 'md',
  className = '',
}: MetricCardProps) {
  const s = sizeClasses[size];
  
  return (
    <div className={`panel ${s.pad} ${className}`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`${s.label} text-[#666] uppercase tracking-wide font-medium`}>{label}</span>
        {icon && <span className="text-[#666]">{icon}</span>}
      </div>
      
      <div className="flex items-baseline gap-2">
        <span className={`${s.value} font-semibold font-mono tabular-nums ${colorClasses[color]}`}>
          {value}
        </span>
        {change !== undefined && (
          <span className={`text-xs font-mono ${change >= 0 ? 'text-green' : 'text-red'}`}>
            {change >= 0 ? '+' : ''}{change.toFixed(2)}%
          </span>
        )}
      </div>
      
      {subValue && <div className="text-xs text-[#666] mt-1">{subValue}</div>}
    </div>
  );
}

/**
 * StatRow
 */
interface StatRowProps {
  children: ReactNode;
  className?: string;
}

export function StatRow({ children, className = '' }: StatRowProps) {
  return (
    <div className={`grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 ${className}`}>
      {children}
    </div>
  );
}

/**
 * MiniMetric
 */
interface MiniMetricProps {
  label: string;
  value: string | number;
  color?: 'green' | 'red' | 'blue' | 'default';
}

export function MiniMetric({ label, value, color = 'default' }: MiniMetricProps) {
  return (
    <div className="text-center">
      <div className={`text-lg font-semibold font-mono tabular-nums ${colorClasses[color]}`}>
        {value}
      </div>
      <div className="text-[10px] text-[#666] uppercase tracking-wide">{label}</div>
    </div>
  );
}
