
import pyvo as vo
import requests
from astropy.io import fits
from astropy.table import Table
from io import BytesIO

def query_irsa_iso(star_name):
    """
    Queries IRSA/SSA for ISO SWS spectra using PyVO.
    Traverses DataLink responses to find FITS data.
    
    Returns
    -------
    list of dict
        List of spectral segments.
    """
    print(f"Searching IRSA (ISO SWS) for {star_name}...")
    segments = []
    
    try:
        # 1. Find SSA Service
        # We look for ISO SWS services. 
        # In testing, we found one valid service.
        services = vo.regsearch(servicetype='ssa', keywords=['ISO', 'SWS'])
        if not services:
            print("  > No ISO SSA service found.")
            return []
            
        svc = services[0] # Take the first one (usually ESA or IRSA)
        
        # 2. Search for Object
        # Need coordinates. Resolve name first? 
        # External query flow usually resolves coords before calling specific modules, 
        # but here we just have star_name.
        # We can use Simbad to resolve or pass coords. 
        # Let's use Simbad resolution for robustness if needed, 
        # but simpler: assume caller might pass coords? 
        # No, query_all passes star_name. 
        # We'll rely on vo.search taking a name if possible, or resolve it.
        # pyvo ssa search typically needs pos (SkyCoord).
        
        from astropy.coordinates import SkyCoord
        try:
             pos = SkyCoord.from_name(star_name)
        except:
             print(f"  > Could not resolve coords for {star_name}")
             return []
             
        res = svc.search(pos=pos, radius=0.01) # Small radius
        
        if len(res) == 0:
            print(f"  > No ISO SWS data found for {star_name}.")
            return []
            
        print(f"  > Found {len(res)} ISO datasets. Fetching best candidate...")
        
        # 3. Process Result (DataLink)
        # We take the first result for now.
        row = res[0]
        datalink_url = row.getdataurl()
        
        # Download DataLink VOTable
        r_link = requests.get(datalink_url, timeout=10)
        if r_link.status_code != 200:
             print("  > Failed to download DataLink.")
             return []
             
        # Parse VOTable
        link_table = Table.read(BytesIO(r_link.content), format='votable')
        
        # Find FITS link
        fits_url = None
        # Look for content_type = application/fits
        if 'content_type' in link_table.colnames and 'access_url' in link_table.colnames:
            for drow in link_table:
                if 'application/fits' in str(drow['content_type']):
                    fits_url = drow['access_url']
                    break
        
        if not fits_url:
             # Fallback: take first access_url
             fits_url = link_table[0]['access_url']
             
        if not fits_url:
             print("  > No FITS URL found in DataLink.")
             return []
             
        print(f"  > Downloading FITS: {fits_url}...")
        
        # 4. Download FITS
        r_fits = requests.get(fits_url, timeout=30)
        if r_fits.status_code != 200:
             print("  > Download failed.")
             return []
             
        # 5. Parse FITS
        with fits.open(BytesIO(r_fits.content)) as hdul:
            # Usually data is in Ext 1 (Binary Table)
            if len(hdul) > 1:
                data = hdul[1].data
                cols = hdul[1].columns.names
                
                # Heuristic for columns
                # ISO SWS often has: 'WAVE', 'FLUX' or similar
                wave_col = next((c for c in cols if 'WAVE' in c or 'LAMBDA' in c), None)
                flux_col = next((c for c in cols if 'FLUX' in c), None)
                
                if wave_col and flux_col:
                    wave = data[wave_col]
                    flux = data[flux_col]
                    
                    # Store
                    segments.append({
                        'wavelength': wave * u.um, # ISO usually microns
                        'flux': flux * u.Jy,       # ISO usually Jy
                        'source': "ISO SWS"
                    })
                    print("  > Spectrum extracted successfully.")
                else:
                    print(f"  > Columns not recognized: {cols}")
            else:
                print("  > FITS has no extensions.")
                
    except Exception as e:
        print(f"  > ISO Query Error: {e}")
        
    return segments
