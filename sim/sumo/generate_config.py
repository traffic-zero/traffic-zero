"""
Generate SUMO configuration files for scenarios.

This script creates .sumocfg files needed for CARLA-SUMO co-simulation.
"""

import os
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def generate_sumocfg(simulation_name: str):
    """
    Generate a .sumocfg file for a given simulation.
    
    Args:
        simulation_name: Name of the simulation folder
    """
    base_dir = Path(__file__).parent.parent / "intersections" / simulation_name
    
    if not base_dir.exists():
        print(f"✗ Simulation directory not found: {base_dir}")
        return False
    
    # Expected files
    net_file = base_dir / "network.net.xml"
    route_file = base_dir / "routes.rou.xml"
    tls_file = base_dir / "tls.add.xml"
    
    # Check required files
    if not net_file.exists():
        print(f"✗ Network file not found: {net_file}")
        print("Run netconvert first or ensure network.net.xml exists")
        return False
    
    if not route_file.exists():
        print(f"⚠ Warning: Routes file not found: {route_file}")
    
    # Create sumocfg XML
    config = ET.Element('configuration')
    
    # Input section
    input_section = ET.SubElement(config, 'input')
    
    net_element = ET.SubElement(input_section, 'net-file')
    net_element.set('value', 'network.net.xml')
    
    if route_file.exists():
        route_element = ET.SubElement(input_section, 'route-files')
        route_element.set('value', 'routes.rou.xml')
    
    if tls_file.exists():
        tls_element = ET.SubElement(input_section, 'additional-files')
        tls_element.set('value', 'tls.add.xml')
    
    # Time section
    time_section = ET.SubElement(config, 'time')
    
    begin_element = ET.SubElement(time_section, 'begin')
    begin_element.set('value', '0')
    
    end_element = ET.SubElement(time_section, 'end')
    end_element.set('value', '3600')  # 1 hour default
    
    step_element = ET.SubElement(time_section, 'step-length')
    step_element.set('value', '0.05')  # Match CARLA step length
    
    # Processing section
    processing = ET.SubElement(config, 'processing')
    
    # Enable lateral resolution for better CARLA sync
    lateral = ET.SubElement(processing, 'lateral-resolution')
    lateral.set('value', '0.8')
    
    # Report section
    report = ET.SubElement(config, 'report')
    
    verbose = ET.SubElement(report, 'verbose')
    verbose.set('value', 'true')
    
    no_warnings = ET.SubElement(report, 'no-warnings')
    no_warnings.set('value', 'false')
    
    # Write to file
    output_file = base_dir / f"{simulation_name}.sumocfg"
    
    tree = ET.ElementTree(config)
    ET.indent(tree, space='    ')
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"✓ Configuration file created: {output_file}")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate SUMO configuration files'
    )
    parser.add_argument(
        'simulation',
        help='Simulation name (e.g., simple4)'
    )
    
    args = parser.parse_args()
    
    success = generate_sumocfg(args.simulation)
    
    if success:
        print("\n✓ Ready for CARLA co-simulation!")
        print(f"\nRun with:")
        print(f"  python -m sim.carla {args.simulation}")
    else:
        print("\n✗ Failed to generate configuration")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

