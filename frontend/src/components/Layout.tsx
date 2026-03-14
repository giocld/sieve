/**
 * Layout - Clean trading dashboard style
 */

import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';
import type { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
  season: string;
  onSeasonChange: (season: string) => void;
  seasons: string[];
}

const navItems = [
  { path: '/', label: 'Overview', icon: HomeIcon },
  { path: '/players', label: 'Players', icon: UsersIcon },
  { path: '/teams', label: 'Teams', icon: BuildingIcon },
  { path: '/similarity', label: 'Similarity', icon: SearchIcon },
];

export function Layout({ children, season, onSeasonChange, seasons }: LayoutProps) {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#0d0d0d] flex flex-col">
      {/* Nav */}
      <nav className="sticky top-0 z-50 bg-[#0d0d0d] border-b border-[#2a2a2a]">
        <div className="container-centered">
          <div className="flex items-center justify-between h-12">
            <Link to="/" className="flex items-center">
              <span className="font-semibold text-[#e5e5e5]">Sieve</span>
            </Link>

            {/* Desktop Nav */}
            <div className="hidden md:flex items-center gap-1">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors ${
                      isActive 
                        ? 'bg-[#1a1a1a] text-[#e5e5e5]' 
                        : 'text-[#666] hover:text-[#e5e5e5] hover:bg-[#141414]'
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>

            {/* Right */}
            <div className="flex items-center gap-2">
              <select
                value={season}
                onChange={(e) => onSeasonChange(e.target.value)}
                className="bg-[#141414] border border-[#2a2a2a] rounded px-2 py-1 text-sm text-[#e5e5e5] focus:outline-none focus:border-[#3b82f6]"
              >
                {seasons.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>

              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-1.5 text-[#666] hover:text-[#e5e5e5]"
              >
                <MenuIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Nav */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-[#2a2a2a] bg-[#0d0d0d]">
            <div className="px-4 py-2 space-y-1">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center gap-3 px-3 py-2 rounded text-sm ${
                      isActive ? 'bg-[#1a1a1a] text-[#e5e5e5]' : 'text-[#666] hover:text-[#e5e5e5]'
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        )}
      </nav>

      {/* Main */}
      <main className="flex-1">
        <div className="container-centered py-6">{children}</div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#2a2a2a]">
        <div className="container-centered py-3">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-[#666]">
            <div>
              <span className="text-[#999]">Sieve Analytics</span>
              <span className="mx-2">|</span>
              <span>NBA Player Value & Efficiency Analysis</span>
            </div>
            <div className="flex items-center gap-4">
              <a href="https://github.com/giocld/sieve" target="_blank" rel="noopener noreferrer" className="hover:text-[#e5e5e5]">
                GitHub
              </a>
              <span>Built with React + FastAPI</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Icons
function HomeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
    </svg>
  );
}

function UsersIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
    </svg>
  );
}

function BuildingIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
    </svg>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  );
}

function MenuIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  );
}

/**
 * PageHeader
 */
interface PageHeaderProps {
  title: string;
  subtitle?: string;
  children?: ReactNode;
}

export function PageHeader({ title, subtitle, children }: PageHeaderProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-2">
      <div>
        <h1 className="text-xl font-semibold text-[#e5e5e5]">{title}</h1>
        {subtitle && <p className="text-sm text-[#666] mt-0.5">{subtitle}</p>}
      </div>
      {children && <div className="flex items-center gap-3">{children}</div>}
    </div>
  );
}

/**
 * Panel components
 */
interface PanelProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}

export function Panel({ children, className = '', onClick }: PanelProps) {
  return (
    <div className={`panel ${className}`} onClick={onClick}>
      {children}
    </div>
  );
}

export function PanelHeader({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`panel-header ${className}`}>{children}</div>;
}

export function PanelBody({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`panel-body ${className}`}>{children}</div>;
}

// Aliases
export const Card = Panel;
export const CardHeader = PanelHeader;
export const CardBody = PanelBody;
