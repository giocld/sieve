import pandas as pd
import re
import sys

def parse_lebron_txt(input_file, output_file, season='2025-26'):
    """
    Parses a text file containing copy-pasted LEBRON data from BBall Index.
    
    Supports two formats:
    1. NEW (2025+): nba_id, Player, Seasons, Team, Pos, MPG, LEBRON, O-LEBRON, D-LEBRON, WAR, OffRole, DefRole
    2. OLD: Rank, Name, Age, Team, Minutes, Role, OffArch, DefRole, WAR, LEBRON, O-LEBRON, D-LEBRON
    """
    print(f"Reading from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print("Empty file!")
        return None
    
    # Check header to determine format
    header = lines[0].strip().lower()
    
    if 'nba_id' in header or 'seasons' in header:
        print("Detected: NEW BBall Index format (2025+)")
        return parse_new_format(lines, output_file, season)
    else:
        print("Detected: OLD BBall Index format")
        return parse_old_format(lines, output_file, season)


def parse_new_format(lines, output_file, season='2025-26'):
    """
    Parse the new BBall Index format (2025+):
    nba_id, Player, Seasons, Team, Pos, MPG, LEBRON, O-LEBRON, D-LEBRON, WAR, OffRole, DefRole
    """
    data = []
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        parts = [p.strip() for p in line.split('\t')]
        
        if len(parts) < 12:
            print(f"Skipping line (not enough columns): {line[:50]}...")
            continue
        
        try:
            nba_id = int(parts[0]) if parts[0].isdigit() else None
            player = parts[1]
            season_year = parts[2]  # e.g., "2026"
            team = parts[3]
            position = parts[4]
            mpg = float(parts[5])
            lebron = float(parts[6])
            o_lebron = float(parts[7])
            d_lebron = float(parts[8])
            war = float(parts[9])
            off_role = parts[10]
            def_role = parts[11]
            
            # Convert season year to season string (e.g., "2026" -> "2025-26")
            if season_year.isdigit():
                year = int(season_year)
                season_str = f"{year-1}-{str(year)[2:]}"
            else:
                season_str = season
            
            # Estimate total minutes from MPG (assume ~70 games played on average)
            estimated_minutes = int(mpg * 70)
            
            entry = {
                'Player': player,
                'PLAYER_ID': nba_id,
                'Age': None,  # Not provided in new format
                'Team(s)': team,
                'Position': position,
                'Minutes': estimated_minutes,
                'MPG': mpg,
                'Rotation Role': None,  # Not provided in new format
                'Offensive Archetype': off_role,
                'Defensive Role': def_role,
                'LEBRON WAR': war,
                'LEBRON': lebron,
                'O-LEBRON': o_lebron,
                'D-LEBRON': d_lebron,
                'Season': season_str
            }
            data.append(entry)
            
        except (ValueError, IndexError) as e:
            print(f"Parse error for line: {line[:50]}... ({e})")
            continue
    
    if not data:
        print("No valid data found!")
        return None
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully converted {len(df)} rows to {output_file}")
    print(f"\nTop 5 players by LEBRON:")
    print(df.nlargest(5, 'LEBRON')[['Player', 'Team(s)', 'LEBRON', 'O-LEBRON', 'D-LEBRON', 'LEBRON WAR']])
    
    return df


def parse_old_format(lines, output_file, season='2025-26'):
    """
    Parse the old BBall Index format:
    Rank, Name, Age, Team, Minutes, Role, OffArch, DefRole, WAR, LEBRON, O-LEBRON, D-LEBRON
    """
    data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try tab-separated first
        parts = [p.strip() for p in line.split('\t') if p.strip()]
        
        if len(parts) >= 12:
            try:
                # Skip rank if present
                if parts[0].isdigit():
                    parts = parts[1:]
                
                name = parts[0]
                age = int(parts[1])
                team = parts[2]
                minutes = int(parts[3])
                role = parts[4]
                off_arch = parts[5]
                def_role = parts[6]
                war = float(parts[7])
                lebron = float(parts[8])
                o_lebron = float(parts[9])
                d_lebron = float(parts[10])
                
                entry = {
                    'Player': name,
                    'PLAYER_ID': None,
                    'Age': age,
                    'Team(s)': team,
                    'Position': None,
                    'Minutes': minutes,
                    'MPG': None,
                    'Rotation Role': role,
                    'Offensive Archetype': off_arch,
                    'Defensive Role': def_role,
                    'LEBRON WAR': war,
                    'LEBRON': lebron,
                    'O-LEBRON': o_lebron,
                    'D-LEBRON': d_lebron,
                    'Season': season
                }
                data.append(entry)
                continue
            except (ValueError, IndexError) as e:
                print(f"Tab parse failed for: {line[:50]}... ({e})")
        
        # Fallback: regex-based parsing for concatenated text
        all_floats = re.findall(r'-?\d{1,2}\.\d{2}', line)
        
        if len(all_floats) < 4:
            continue
        
        war = float(all_floats[-4])
        lebron = float(all_floats[-3])
        o_lebron = float(all_floats[-2])
        d_lebron = float(all_floats[-1])
        
        first_float_str = all_floats[-4]
        float_start_index = line.rfind(first_float_str)
        
        if float_start_index == -1:
            continue
        
        remaining = line[:float_start_index].strip()
        
        def_roles = ['Helper', 'Chaser', 'Point of Attack', 'Wing Stopper', 'Anchor Big', 
                    'Mobile Big', 'Low Activity']
        off_archs = ['Shot Creator', 'Movement Shooter', 'Primary Ball Handler', 'Athletic Finisher',
                    'Stretch Big', 'Post Scorer', 'Slasher', 'Off Screen Shooter', 
                    'Secondary Ball Handler', 'Versatile Big', 'Roll + Cut Big', 'Low Minute',
                    'Stationary Shooter']
        roles = ['Star', 'Starter', 'Key Rotation', 'Rotation', 'Garbage Time', 'Too Few Games']
        
        def_role = None
        for dr in def_roles:
            if remaining.endswith(dr):
                def_role = dr
                remaining = remaining[:-len(dr)].strip()
                break
        
        off_arch = None
        for oa in off_archs:
            if remaining.endswith(oa):
                off_arch = oa
                remaining = remaining[:-len(oa)].strip()
                break
        
        role = None
        for r in roles:
            if remaining.endswith(r):
                role = r
                remaining = remaining[:-len(r)].strip()
                break
        
        if not all([def_role, off_arch]):
            continue
        
        int_match = re.search(r'(\d+)$', remaining)
        if int_match:
            minutes = int(int_match.group(1))
            remaining = remaining[:int_match.start()].strip()
        else:
            continue
        
        team_match = re.search(r'([A-Z]{2,3}(?:/[A-Z]{2,3})?)$', remaining)
        if team_match:
            team = team_match.group(1)
            remaining = remaining[:team_match.start()].strip()
        else:
            continue
        
        age_match = re.search(r'(\d{2})$', remaining)
        if age_match:
            age = int(age_match.group(1))
            remaining = remaining[:age_match.start()].strip()
        else:
            continue
        
        name = re.sub(r'^\d+\s*', '', remaining).strip()
        
        if not name:
            continue
        
        entry = {
            'Player': name,
            'PLAYER_ID': None,
            'Age': age,
            'Team(s)': team,
            'Position': None,
            'Minutes': minutes,
            'MPG': None,
            'Rotation Role': role,
            'Offensive Archetype': off_arch,
            'Defensive Role': def_role,
            'LEBRON WAR': war,
            'LEBRON': lebron,
            'O-LEBRON': o_lebron,
            'D-LEBRON': d_lebron,
            'Season': season
        }
        data.append(entry)

    if not data:
        print("No valid data found!")
        return None

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Successfully converted {len(df)} rows to {output_file}")
    print(f"\nTop 5 players:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    input_path = 'data/lebroninput.txt'
    output_path = 'data/LEBRON_2025-26.csv'
    
    df = parse_lebron_txt(input_path, output_path)
    
    if df is not None:
        from src.cache_manager import cache
        cache.save_lebron_metrics(df, season='2025-26')
        print(f"\nSaved to database as well!")
