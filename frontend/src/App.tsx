/**
 * Sieve NBA Analytics - Main Application
 */

import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { Layout } from './components/Layout';
import { Overview, Players, Teams, Similarity } from './pages';
import { useSeasons } from './hooks/useApi';
import { PageLoading, ErrorDisplay } from './components/Loading';

// =============================================================================
// CACHING STRATEGY
// Data loads instantly from localStorage, refreshes in background
// =============================================================================

const THIRTY_MINUTES = 1000 * 60 * 30;
const TWO_HOURS = 1000 * 60 * 60 * 2;
const CACHE_KEY = 'sieve-query-cache';
const CACHE_VERSION = 'v4'; // Bump this to invalidate old cached data

const IS_DEV = import.meta.env.DEV;

// Create React Query client. In production we cache aggressively for fast loads;
// in dev we always revalidate so refreshed backend data shows up immediately.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: IS_DEV ? 0 : THIRTY_MINUTES,
      gcTime: TWO_HOURS,
      refetchOnWindowFocus: IS_DEV,
      refetchOnReconnect: IS_DEV,
      retry: 2,
    },
  },
});

// Save cache to localStorage
function saveCache() {
  try {
    const cache = queryClient.getQueryCache().getAll();
    const serializable = cache
      .filter(query => query.state.data !== undefined)
      .map(query => ({
        queryKey: query.queryKey,
        data: query.state.data,
        dataUpdatedAt: query.state.dataUpdatedAt,
      }));

    localStorage.setItem(CACHE_KEY, JSON.stringify({
      version: CACHE_VERSION,
      timestamp: Date.now(),
      queries: serializable,
    }));
  } catch {
    // localStorage might be full or disabled
  }
}

// Restore cache from localStorage on app start
function restoreCache() {
  try {
    const stored = localStorage.getItem(CACHE_KEY);
    if (!stored) return;

    const { version, timestamp, queries } = JSON.parse(stored);

    // Invalidate old cache
    if (version !== CACHE_VERSION || Date.now() - timestamp > TWO_HOURS) {
      localStorage.removeItem(CACHE_KEY);
      return;
    }

    queries.forEach((query: { queryKey: unknown[]; data: unknown; dataUpdatedAt: number }) => {
      queryClient.setQueryData(query.queryKey, query.data, {
        updatedAt: query.dataUpdatedAt,
      });
    });
  } catch {
    localStorage.removeItem(CACHE_KEY);
  }
}

// Initialize cache. Skip localStorage hydration entirely in dev so we never
// serve stale data off disk while iterating on the backend / pipeline.
if (!IS_DEV) {
  restoreCache();
  setInterval(saveCache, 30000);
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', saveCache);
  }
} else if (typeof window !== 'undefined') {
  try {
    localStorage.removeItem(CACHE_KEY);
  } catch {
    // ignore
  }
}


function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

function AppContent() {
  const { data: seasonsData, isLoading, error } = useSeasons();
  const [season, setSeason] = useState<string>('');

  // Set initial season when data loads
  useEffect(() => {
    if (seasonsData?.current && !season) {
      setSeason(seasonsData.current);
    }
  }, [seasonsData?.current, season]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0e14] flex items-center justify-center">
        <PageLoading />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#0a0e14] flex items-center justify-center">
        <ErrorDisplay
          title="Failed to connect to API"
          message="Make sure the backend server is running on port 8000"
        />
      </div>
    );
  }

  const seasons = (seasonsData?.seasons || []).filter(s => s !== '2024-25');

  return (
    <Layout
      season={season}
      onSeasonChange={setSeason}
      seasons={seasons}
    >
      <Routes>
        <Route path="/" element={<Overview season={season} />} />
        <Route path="/players" element={<Players season={season} />} />
        <Route path="/teams" element={<Teams season={season} />} />
        <Route path="/similarity" element={<Similarity season={season} />} />
      </Routes>
    </Layout>
  );
}

export default App;
