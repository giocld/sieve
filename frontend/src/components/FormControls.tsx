/**
 * Form controls - Clean style
 */

import type { InputHTMLAttributes, SelectHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';

// Select
interface SelectOption {
  label: string;
  value: string;
}

interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  options: SelectOption[];
  label?: string;
  onChange: (value: string) => void;
}

export function Select({ options, label, onChange, className = '', ...props }: SelectProps) {
  return (
    <div className="flex flex-col gap-1">
      {label && <label className="text-[11px] text-[#666] uppercase tracking-wide">{label}</label>}
      <select
        {...props}
        onChange={(e) => onChange(e.target.value)}
        className={`bg-[#141414] border border-[#2a2a2a] rounded px-3 py-2 text-sm text-[#e5e5e5] focus:outline-none focus:border-[#3b82f6] cursor-pointer disabled:opacity-50 ${className}`}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

// Input
interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

export function Input({ label, error, leftIcon, rightIcon, className = '', ...props }: InputProps) {
  return (
    <div className="flex flex-col gap-1">
      {label && <label className="text-[11px] text-[#666] uppercase tracking-wide">{label}</label>}
      <div className="relative">
        {leftIcon && <div className="absolute left-3 top-1/2 -translate-y-1/2 text-[#666]">{leftIcon}</div>}
        <input
          {...props}
          className={`w-full bg-[#141414] border border-[#2a2a2a] rounded py-2 ${leftIcon ? 'pl-11 pr-3' : rightIcon ? 'pl-3 pr-10' : 'px-3'} text-sm text-[#e5e5e5] placeholder:text-[#444] focus:outline-none focus:border-[#3b82f6] disabled:opacity-50 ${error ? 'border-[#ef4444]' : ''} ${className}`}
        />
        {rightIcon && <div className="absolute right-3 top-1/2 -translate-y-1/2 text-[#666]">{rightIcon}</div>}
      </div>
      {error && <span className="text-xs text-[#ef4444]">{error}</span>}
    </div>
  );
}

// Button
interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

const variantClasses = {
  primary: 'bg-[#3b7be9] text-white hover:bg-[#2f68d4] font-medium',
  secondary: 'bg-[#1a1a1a] text-[#e5e5e5] hover:bg-[#222] border border-[#2a2a2a]',
  ghost: 'text-[#999] hover:text-[#e5e5e5] hover:bg-[#1a1a1a]',
};

const sizeClasses = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  leftIcon,
  rightIcon,
  children,
  className = '',
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center gap-2 rounded transition-colors disabled:opacity-50 ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
    >
      {loading ? <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" /> : leftIcon}
      {children}
      {rightIcon}
    </button>
  );
}

// Slider
interface SliderProps {
  label?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  formatValue?: (value: number) => string;
  className?: string;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  formatValue = (v) => String(v),
  className = '',
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center justify-between">
        {label && <label className="text-[11px] text-[#666] uppercase tracking-wide">{label}</label>}
        <span className="text-sm font-mono text-[#3b82f6]">{formatValue(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 rounded appearance-none cursor-pointer bg-[#2a2a2a] [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#3b82f6] [&::-webkit-slider-thumb]:cursor-pointer"
        style={{ background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${pct}%, #2a2a2a ${pct}%, #2a2a2a 100%)` }}
      />
      <div className="flex justify-between text-[10px] text-[#444]">
        <span>{formatValue(min)}</span>
        <span>{formatValue(max)}</span>
      </div>
    </div>
  );
}

// RangeSlider
interface RangeSliderProps {
  label?: string;
  minValue: number;
  maxValue: number;
  min: number;
  max: number;
  step?: number;
  onMinChange: (value: number) => void;
  onMaxChange: (value: number) => void;
  formatValue?: (value: number) => string;
  className?: string;
}

export function RangeSlider({
  label,
  minValue,
  maxValue,
  min,
  max,
  step = 1,
  onMinChange,
  onMaxChange,
  formatValue = (v) => String(v),
  className = '',
}: RangeSliderProps) {
  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center justify-between">
        {label && <label className="text-[11px] text-[#666] uppercase tracking-wide">{label}</label>}
        <span className="text-sm text-[#999]">{formatValue(minValue)} - {formatValue(maxValue)}</span>
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          <Slider value={minValue} min={min} max={maxValue - step} step={step} onChange={onMinChange} formatValue={formatValue} />
        </div>
        <div className="flex-1">
          <Slider value={maxValue} min={minValue + step} max={max} step={step} onChange={onMaxChange} formatValue={formatValue} />
        </div>
      </div>
    </div>
  );
}
