import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import Plot from 'react-plotly.js';
import type { Config } from 'plotly.js';
import { PlayerHoverCard } from './PlayerHoverCard';

interface ChartPanelProps {
  title?: string;
  subtitle?: string;
  chartJson?: string;
  isLoading?: boolean;
  error?: Error | null;
  height?: number | string;
  className?: string;
  showHeader?: boolean;
  enablePlayerHover?: boolean;
}

const defaultConfig: Partial<Config> = {
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  responsive: true,
};

export function ChartPanel({
  title,
  subtitle,
  chartJson,
  isLoading = false,
  error = null,
  height = 400,
  className = '',
  showHeader = true,
  enablePlayerHover = true,
}: ChartPanelProps) {
  const [hoverData, setHoverData] = useState<{ player: any; x: number; y: number } | null>(null);
  const [mobileTapData, setMobileTapData] = useState<{ player: any; x: number; y: number } | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const plotDivRef = useRef<any>(null);
  const plotlyClickHandlerRef = useRef<((event: any) => void) | null>(null);

  const chartData = useMemo(() => {
    if (!chartJson) return null;

    try {
      const parsed = JSON.parse(chartJson);

      const data = enablePlayerHover
        ? (parsed.data || []).map((trace: any) => ({
            ...trace,
            hoverinfo: 'none',
            hovertemplate: null,
          }))
        : (parsed.data || []);

      const mergedLayout = {
        ...parsed.layout,
        paper_bgcolor: 'transparent',
        plot_bgcolor: parsed.layout?.plot_bgcolor || '#141414',
        hovermode: 'closest',
        font: {
          ...parsed.layout?.font,
          color: parsed.layout?.font?.color || '#999',
          family: 'Inter, sans-serif',
        },
        margin: parsed.layout?.margin,
        xaxis: {
          ...parsed.layout?.xaxis,
          gridcolor: parsed.layout?.xaxis?.gridcolor || '#2a2a2a',
          linecolor: '#333',
        },
        yaxis: {
          ...parsed.layout?.yaxis,
          gridcolor: parsed.layout?.yaxis?.gridcolor || '#2a2a2a',
          linecolor: '#333',
        },
        height: parsed.layout?.height || (typeof height === 'number' ? height : undefined),
        autosize: true,
      };

      return { data, layout: mergedLayout };
    } catch (e) {
      console.error('Failed to parse chart JSON:', e);
      return null;
    }
  }, [chartJson, height, enablePlayerHover]);

  const handleHover = useCallback((event: Readonly<Plotly.PlotMouseEvent>) => {
    if (!enablePlayerHover) return;

    const point = event.points[0];
    const customdata = point.customdata as any;

    if (customdata && (customdata.player_name || customdata.player_id)) {
      const { clientX, clientY } = event.event as unknown as React.MouseEvent;
      setHoverData({
        player: customdata,
        x: clientX,
        y: clientY,
      });
    }
  }, [enablePlayerHover]);

  const handleUnhover = useCallback(() => {
    if (enablePlayerHover) {
      setHoverData(null);
    }
  }, [enablePlayerHover]);

  const isTouchDevice = useCallback(() => {
    if (typeof window === 'undefined' || typeof navigator === 'undefined') return false;
    const coarsePointer = typeof window.matchMedia === 'function'
      && window.matchMedia('(hover: none), (pointer: coarse)').matches;
    return coarsePointer || navigator.maxTouchPoints > 0 || 'ontouchstart' in window;
  }, []);

  const handleMobilePlotlyClick = useCallback((event: any) => {
    if (!enablePlayerHover || !isTouchDevice()) return;

    const point = event?.points?.[0];
    const customdata = point?.customdata as any;
    if (!(customdata && (customdata.player_name || customdata.player_id))) {
      return;
    }

    const x = Math.round(window.innerWidth / 2);
    const y = Math.max(160, window.innerHeight - 20);

    setMobileTapData({
      player: customdata,
      x,
      y,
    });
  }, [enablePlayerHover, isTouchDevice]);

  const bindMobileClickHandler = useCallback((graphDiv: any) => {
    if (!graphDiv || !graphDiv.on) return;

    if (plotlyClickHandlerRef.current && graphDiv.removeListener) {
      graphDiv.removeListener('plotly_click', plotlyClickHandlerRef.current);
    }

    plotlyClickHandlerRef.current = handleMobilePlotlyClick;
    graphDiv.on('plotly_click', plotlyClickHandlerRef.current);
  }, [handleMobilePlotlyClick]);

  useEffect(() => {
    if (!plotDivRef.current) return;
    bindMobileClickHandler(plotDivRef.current);
  }, [bindMobileClickHandler, chartData]);

  useEffect(() => {
    if (!mobileTapData) return;

    const handleOutsideTouch = (event: TouchEvent) => {
      if (!panelRef.current) return;
      if (!panelRef.current.contains(event.target as Node)) {
        setMobileTapData(null);
      }
    };

    document.addEventListener('touchstart', handleOutsideTouch, { passive: true });
    return () => {
      document.removeEventListener('touchstart', handleOutsideTouch);
    };
  }, [mobileTapData]);

  return (
    <div ref={panelRef} className={`panel overflow-hidden ${className}`}>
      {showHeader && (title || subtitle) && (
        <div className="panel-header">
          {title && <span>{title}</span>}
          {subtitle && <span className="ml-2 text-[#666] font-normal">{subtitle}</span>}
        </div>
      )}

      <div className="relative" style={{ height: typeof height === 'string' ? height : `${height}px` }}>
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#141414]">
            <div className="flex flex-col items-center gap-3">
              <div className="w-6 h-6 border-2 border-[#22c55e] border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-[#666]">Loading chart...</span>
            </div>
          </div>
        )}

        {error && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#141414]">
            <div className="text-center px-4">
              <div className="text-[#ef4444] text-sm mb-1">Error</div>
              <div className="text-xs text-[#666]">{error.message}</div>
            </div>
          </div>
        )}

        {chartData && !isLoading && !error && (
          <Plot
            data={chartData.data}
            layout={chartData.layout}
            config={defaultConfig}
            className="w-full h-full"
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
            onHover={handleHover}
            onUnhover={handleUnhover}
            onInitialized={(_figure, graphDiv) => {
              plotDivRef.current = graphDiv;
              bindMobileClickHandler(graphDiv);
            }}
            onUpdate={(_figure, graphDiv) => {
              plotDivRef.current = graphDiv;
              bindMobileClickHandler(graphDiv);
            }}
          />
        )}

        {!chartData && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#141414]">
            <span className="text-xs text-[#666]">No data</span>
          </div>
        )}
      </div>

      {hoverData && (
        <PlayerHoverCard
          player={hoverData.player}
          manualPosition={{ x: hoverData.x, y: hoverData.y }}
        />
      )}

      {mobileTapData && (
        <PlayerHoverCard
          player={mobileTapData.player}
          manualPosition={{ x: mobileTapData.x, y: mobileTapData.y }}
        />
      )}
    </div>
  );
}

/**
 * ChartGrid
 */
interface ChartGridProps {
  children: React.ReactNode;
  cols?: 1 | 2 | 3;
  className?: string;
}

export function ChartGrid({ children, cols = 2, className = '' }: ChartGridProps) {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 lg:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
  };

  return <div className={`grid ${gridCols[cols]} gap-4 ${className}`}>{children}</div>;
}
