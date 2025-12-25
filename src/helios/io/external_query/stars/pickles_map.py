
# Pickles Library (1998) Map
# Maps Spectral Type (upper case) to filename in J/PASP/110/863
# Source: Vizier J/PASP/110/863/table1

PICKLES_MAP = {
    # O Stars
    'O5V': 'uk05v', 'O9V': 'uk09v',
    
    # B Stars
    'B0V': 'ukb0v', 'B1V': 'ukb1v', 'B3V': 'ukb3v', 'B5V': 'ukb57v', 'B8V': 'ukb8v', 'B9V': 'ukb9v',
    'B0I': 'ukb0i', 'B1I': 'ukb1i', 'B3I': 'ukb3i', 'B5I': 'ukb5i', 'B8I': 'ukb8i',
    
    # A Stars
    'A0V': 'uka0v', 'A2V': 'uka2v', 'A3V': 'uka3v', 'A5V': 'uka5v', 'A7V': 'uka7v',
    'A0I': 'uka0i', 'A2I': 'uka2i',
    
    # F Stars
    'F0V': 'ukf0v', 'F2V': 'ukf2v', 'F5V': 'ukf5v', 'F8V': 'ukf8v',
    'F0I': 'ukf0i', 'F2I': 'ukf2i', 'F5I': 'ukf5i', 'F8I': 'ukf8i',
    
    # G Stars
    'G0V': 'ukg0v', 'G2V': 'ukg2v', 'G5V': 'ukg5v', 'G8V': 'ukg8v',
    'G0I': 'ukg0i', 'G2I': 'ukg2i', 'G5I': 'ukg5i', 'G8I': 'ukg8i',
    
    # K Stars
    'K0V': 'ukk0v', 'K2V': 'ukk2v', 'K3V': 'ukk3v', 'K4V': 'ukk4v', 'K5V': 'ukk5v', 'K7V': 'ukk7v',
    'K0I': 'ukk0i', 'K2I': 'ukk2i', 'K3I': 'ukk3i', 'K4I': 'ukk4i', 'K5I': 'ukk5m', # ukk5m is K5I in Pickles logic sometimes? Or use closest.
    
    # M Stars (Dwarfs)
    'M0V': 'ukm0v', 'M1V': 'ukm1v', 'M2V': 'ukm2v', 'M3V': 'ukm3v', 'M4V': 'ukm4v', 'M5V': 'ukm5v', 'M6V': 'ukm6v',
    
    # M Stars (Giants/Supergiants - Pickles uses I/II/III mixed potentially, check specifics)
    # Betelgeuse is M2Iab. Pickles has 'ukm2i'
    'M0I': 'ukm0i', 'M1I': 'ukm1i', 'M2I': 'ukm2i', 'M3I': 'ukm3i', 'M4I': 'ukm4i',
    
    # Giants (III) - often default in Pickles if not specified as V or I? 
    # Actually Pickles has distinct III files. e.g. 'ukg8iii'
    'G8III': 'ukg8iii', 'K0III': 'ukk0iii', 'K2III': 'ukk2iii', 'M0III': 'ukm0iii', 'M2III': 'ukm2iii'
}

def normalize_sptype(sptype):
    """
    Attempts to normalize input spectral type to Pickles equivalent key.
    """
    if not sptype: return None
    s = sptype.upper().replace(' ', '')
    
    # Handle 'Iab' -> 'I'
    if 'IAB' in s: s = s.replace('IAB', 'I')
    elif 'IA' in s: s = s.replace('IA', 'I')
    elif 'IB' in s: s = s.replace('IB', 'I')
    
    # Handle decimals: M2.5V -> M2V or M3V? 
    # Pickles doesn't have 2.5. We round.
    if '.' in s:
        # Crude rounding
        import re
        match = re.match(r'([A-Z])(\d+)\.?(\d+)?(.*)', s)
        if match:
            let, num, dec, lum = match.groups()
            if dec:
                # if >= 5 round up
                if int(dec[0]) >= 5: n = int(num) + 1
                else: n = int(num)
                s = f"{let}{n}{lum}"
            
    return s
