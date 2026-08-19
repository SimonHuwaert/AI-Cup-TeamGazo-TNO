import pandas as pd
import struct

def decode_robin_trajectory(hex_str):
    """
    Decodes WKB hex string for LineString ZM (SRID 4326).
    Returns a string of points: (Lon, Lat, Alt, M), (Lon, Lat, Alt, M)...
    """
    if not isinstance(hex_str, str) or len(hex_str) < 26:
        return ""
    
    try:
        # Convert hex string to raw bytes
        binary_data = bytes.fromhex(hex_str)
        
        # Byte 0: Endianness (01 = Little Endian)
        endian = '<' if binary_data[0] == 1 else '>'
        
        # Points count starts at byte 9 (after 1 byte endian, 4 bytes type, 4 bytes SRID)
        num_points = struct.unpack(endian + 'I', binary_data[9:13])[0]
        
        coords = []
        offset = 13
        # Each point consists of 4 doubles (8 bytes each): X, Y, Z, M
        for _ in range(num_points):
            # Unpack 4 doubles (32 bytes total)
            point = struct.unpack(endian + 'dddd', binary_data[offset : offset + 32])
            # Format as (X, Y, Z, M)
            coords.append(f"({point[0]}, {point[1]}, {point[2]}, {point[3]})")
            offset += 32
            
        return ", ".join(coords)
    
    except Exception as e:
        return f"Error: {str(e)}"



# 1. Define your file path
file_path_1 = r"test.csv"
file_path_2 = r"train.csv"

# 2. Load the CSV
df_1 = pd.read_csv(file_path_1)
df_2 = pd.read_csv(file_path_2)

# 3. Process the 'trajectory' column in the test file
if 'trajectory' in df_1.columns:
    print("Decoding trajectories...")
    df_1['trajectory'] = df_1['trajectory'].apply(decode_robin_trajectory)
    
    # 4. Save the results
    df_1.to_csv(file_path_1, index=False)
    print(f"Success! File saved to: {file_path_1}")
else:
    print("Error: Could not find a column named 'trajectory' in the CSV.")


# 3. Process the 'trajectory' column in the train file
if 'trajectory' in df_2.columns:
    print("Decoding trajectories...")
    df_2['trajectory'] = df_2['trajectory'].apply(decode_robin_trajectory)
    
    # 4. Save the results
    df_2.to_csv(file_path_2, index=False)
    print(f"Success! File saved to: {file_path_2}")
else:
    print("Error: Could not find a column named 'trajectory' in the CSV.")