/**
 * Loading components - Clean style
 */

import type { ReactNode } from 'react';

interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4 border',
  md: 'w-6 h-6 border-2',
  lg: 'w-8 h-8 border-2',
};

export function Loading({ size = 'md', className = '' }: LoadingProps) {
  return (
    <div className={`${sizeClasses[size]} border-[#22c55e] border-t-transparent rounded-full animate-spin ${className}`} />
  );
}

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[#0d0d0d]/90 z-50">
      <div className="flex flex-col items-center gap-3">
        <Loading size="lg" />
        <span className="text-sm text-[#999]">{message}</span>
      </div>
    </div>
  );
}

interface SkeletonProps {
  className?: string;
  children?: ReactNode;
}

export function Skeleton({ className = '', children }: SkeletonProps) {
  return (
    <div className={`skeleton ${className}`}>
      {children && <span className="invisible">{children}</span>}
    </div>
  );
}

export function PageLoading() {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="flex flex-col items-center gap-4">
        <Loading size="lg" />
        <p className="text-[#999] text-sm">Loading data...</p>
      </div>
    </div>
  );
}

interface ErrorDisplayProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export function ErrorDisplay({ title = 'Error', message, onRetry }: ErrorDisplayProps) {
  return (
    <div className="flex items-center justify-center min-h-[200px]">
      <div className="text-center">
        <div className="text-[#ef4444] text-sm font-medium mb-2">{title}</div>
        <p className="text-sm text-[#666] mb-4">{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-2 bg-[#1a1a1a] text-[#e5e5e5] text-sm rounded hover:bg-[#222] transition-colors border border-[#2a2a2a]"
          >
            Try Again
          </button>
        )}
      </div>
    </div>
  );
}
