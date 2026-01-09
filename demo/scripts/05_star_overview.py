
from helios.io.external_query.stars.query_all import get_star_properties
from helios.utils.data_completion.star import overview

def run_demo(save=False):
    print("--- Fetching Vega Data ---")
    data = get_star_properties("Vega", complete_data=False, plot=False)
    
    print("\n--- Star Data Overview ---")
    # Overview handles printing, no return value to check or plot to save here.
    overview(data)
    
    # Because this demo only prints to console, save param is ignored/unused.
    if save:
        print("Note: This demo only prints text output, no plots to save.")

if __name__ == "__main__":
    run_demo()
