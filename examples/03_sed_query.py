
"""
Demo script for retrieving comprehensive Star Properties using helios.
"""

import matplotlib.pyplot as plt
from helios.io.external_query.stars.query_all import get_star_properties
import pprint

def main():
    stars = ["Vega", "Betelgeuse", "Sirius"]
    
    plt.figure(figsize=(10, 6))
    
    for star in stars:
        print(f"\n--- Processing {star} ---")
        try:
            data = get_star_properties(star, complete_data=True, plot=False)
            if data:
                # Pretty print the dictionary structure
                print(f"Data retrieved for {star}")
                # pprint.pprint(data, depth=2)
                
                sed = data['sed']
                photo = data.get('photometry', {})
                
                # Plot High-Res Model
                if len(sed['wavelength']) > 0:
                    line, = plt.loglog(sed['wavelength'], sed['flux'], '-', label=f"{star} Model")
                    color = line.get_color() # Get color of the model
                else:
                    color = 'blue' # default
                
                # Plot Photometry if available
                if 'wavelength' in photo and len(photo['wavelength']) > 0:
                    # Use same color as model
                    # Use errorbar safely
                    yerr = photo.get('flux_error', None)
                    plt.errorbar(photo['wavelength'], photo['flux'], 
                                 yerr=yerr, 
                                 fmt='o', color=color, ecolor=color, 
                                 label=f"{star} Photometry", markersize=4, capsize=3)
            else:
                print("  No data found.")
        except Exception as e:
            print(f"  Error: {e}")
            
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Flux (Jy)')
    plt.title('Multi-Star SED Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    output_file = "sed_demo_plot.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
