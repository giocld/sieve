/**
 * Sieve NBA Analytics - Main Application
 */

import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { Layout } from './components/Layout';
import { Overview, Players, Teams, Lineups, Similarity } from './pages';
import { useSeasons } from './hooks/useApi';
import { PageLoading, ErrorDisplay } from './components/Loading';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

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

  const seasons = seasonsData?.seasons || [];

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
        <Route path="/lineups" element={<Lineups />} />
        <Route path="/similarity" element={<Similarity season={season} />} />
      </Routes>
    </Layout>
  );
}

export default App;
