import pandas as pd
import re
import sys

def parse_lebron_txt(input_file, output_file, season='2025-26'):
    """
    Parses a text file containing copy-pasted LEBRON data.
    Works backwards from the end to extract numeric fields.
    """
    print(f"Reading from {input_file}...")
    
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try tab-separated first (proper format)
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            
            if len(parts) >= 12:
                # Tab-separated format: Rank, Name, Age, Team, Minutes, Role, OffArch, DefRole, WAR, LEBRON, O-LEBRON, D-LEBRON
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
                        'Age': age,
                        'Team(s)': team,
                        'Minutes': minutes,
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
                    continue # Successfully parsed with tabs, move to next line
                except (ValueError, IndexError) as e:
                    print(f"Tab parse failed for: {line[:50]}... ({e})")
            
            # Fallback: concatenated format (no tabs preserved)
            # Find all floats in the line using pattern for X.XX format
            all_floats = re.findall(r'-?\d{1,2}\.\d{2}', line)
            
            if len(all_floats) < 4:
                print(f"Could not find 4 floats in '{line[:50]}...'")
                continue
            
            # Take the last 4 floats
            war = float(all_floats[-4])
            lebron = float(all_floats[-3])
            o_lebron = float(all_floats[-2])
            d_lebron = float(all_floats[-1])
            
            # Extract text part before the floats
            # Find the starting position of the first of the last four floats
            first_float_str = all_floats[-4]
            float_start_index = line.rfind(first_float_str) # Use rfind to get the last occurrence
            
            if float_start_index != -1:
                remaining = line[:float_start_index].strip()
            else:
                # This case should ideally not happen if all_floats[-4] was found
                print(f"Error: Could not locate start of floats in line: {line[:50]}...")
                continue
            
            # Now remaining should end with: def_role, off_arch, role, minutes, team, age, name
            # The def_role and off_arch are text fields that run together
            # Let's look for known patterns
            
            # Common defensive roles
            def_roles = ['Helper', 'Chaser', 'Point of Attack', 'Wing Stopper', 'Anchor Big', 
                        'Mobile Big', 'Low Activity']
            # Common offensive archetypes
            off_archs = ['Shot Creator', 'Movement Shooter', 'Primary Ball Handler', 'Athletic Finisher',
                        'Stretch Big', 'Post Scorer', 'Slasher', 'Off Screen Shooter', 
                        'Secondary Ball Handler', 'Versatile Big', 'Roll + Cut Big', 'Low Minute']
            # Rotation roles
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
            
            if not all([def_role, off_arch, role]):
                print(f"Could not parse roles from: {line[:50]}...")
                continue
            
            # Now extract minutes (last integer)
            int_match = re.search(r'(\d+)$', remaining)
            if int_match:
                minutes = int(int_match.group(1))
                remaining = remaining[:int_match.start()].strip()
            else:
                print(f"Could not extract minutes from: {remaining[:50]}...")
                continue
            
            # Extract team (3 letter code or 3-4 letter like SAC/DET)
            team_match = re.search(r'([A-Z]{2,3}(?:/[A-Z]{2,3})?)$', remaining)
            if team_match:
                team = team_match.group(1)
                remaining = remaining[:team_match.start()].strip()
            else:
                print(f"Could not extract team from: {remaining[:50]}...")
                continue
            
            # Extract age (last 2-digit number)
            age_match = re.search(r'(\d{2})$', remaining)
            if age_match:
                age = int(age_match.group(1))
                remaining = remaining[:age_match.start()].strip()
            else:
                print(f"Could not extract age from: {remaining[:50]}...")
                continue
            
            # Remaining is: rank + name or just name
            # Remove leading rank number if present
            name = re.sub(r'^\d+\s*', '', remaining).strip()
            
            if not name:
                print(f"Could not extract name from: {line[:50]}...")
                continue
            
            entry = {
                'Player': name,
                'Age': age,
                'Team(s)': team,
                'Minutes': minutes,
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
    
    # Save to CSV
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
        # Also save to database
        from src.cache_manager import cache
        cache.save_lebron_metrics(df, season='2025-26')
        print(f"\nSaved to database as well!")
