
from helios.io.external_query.stars.query_all import get_star_properties
from helios.utils.data_completion.star import overview

def main():
    print("--- Fetching Vega Data ---")
    data = get_star_properties("Vega", complete_data=False, plot=False)
    
    print("\n--- Star Data Overview ---")
    overview(data)

if __name__ == "__main__":
    main()
