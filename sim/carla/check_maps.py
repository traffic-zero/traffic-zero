"""Check available CARLA maps."""
import carla

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    
    maps = client.get_available_maps()
    
    print("Available CARLA maps:")
    print("=" * 60)
    for i, map_name in enumerate(maps, 1):
        print(f"{i}. {map_name}")
    
    print("\n" + "=" * 60)
    print("Looking for empty/plain maps:")
    empty_maps = [m for m in maps if 'empty' in m.lower()]
    if empty_maps:
        print("Empty maps found:")
        for m in empty_maps:
            print(f"  - {m}")
    else:
        print("No maps with 'empty' in the name found.")
        print("\nRecommended minimal maps:")
        minimal = [m for m in maps if any(x in m.lower() for x in ['town01', 'town02', 'town03'])]
        for m in minimal[:3]:
            print(f"  - {m}")
            
except Exception as e:
    print(f"Error: {e}")
    print("Make sure CARLA is running!")

