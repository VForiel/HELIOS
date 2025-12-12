"""
Convert colored SVG icons to black and white for theme adaptation.
This script replaces all fill and stroke colors with black (#000000).
"""

import os
import re
from pathlib import Path

def convert_svg_to_bw(svg_content):
    """Convert SVG content to black and white by replacing colors with black."""
    
    # Replace all fill colors with black
    svg_content = re.sub(r'fill="[^"]*"', 'fill="#000000"', svg_content)
    
    # Replace all stroke colors with black (except "none")
    def replace_stroke(match):
        color = match.group(1)
        if color.lower() == 'none':
            return match.group(0)
        return 'stroke="#000000"'
    
    svg_content = re.sub(r'stroke="([^"]*)"', replace_stroke, svg_content)
    
    # Remove opacity attributes to ensure solid black
    svg_content = re.sub(r'\s+opacity="[^"]*"', '', svg_content)
    
    return svg_content

def main():
    icons_dir = Path(__file__).parent.parent / 'web' / 'frontend' / 'public' / 'icons'
    
    if not icons_dir.exists():
        print(f"Icons directory not found: {icons_dir}")
        return
    
    print(f"Converting SVG icons in: {icons_dir}")
    
    converted_count = 0
    for svg_file in icons_dir.glob('*.svg'):
        try:
            # Read original content
            with open(svg_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Convert to black and white
            bw_content = convert_svg_to_bw(original_content)
            
            # Write back
            with open(svg_file, 'w', encoding='utf-8') as f:
                f.write(bw_content)
            
            print(f"✓ Converted: {svg_file.name}")
            converted_count += 1
            
        except Exception as e:
            print(f"✗ Error converting {svg_file.name}: {e}")
    
    print(f"\nConverted {converted_count} SVG files to black and white.")

if __name__ == '__main__':
    main()
