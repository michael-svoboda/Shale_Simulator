import json
import numpy as np
import struct
import base64
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Primitive,
    Animation, AnimationChannel, AnimationSampler, BufferFormat
)

###############################################################################
# 1. Configuration and Utilities
###############################################################################
visualScale = 0.2      # scales the cell dimensions (dx, dy)
base_thickness = 0.1   # minimum block height

def pressure_to_color(p):
    """
    Maps a pressure value (in Pa) to a color.
    25e6 Pa => purple, 35e6 Pa => red.
    Intermediate stops:
      0.0: purple (0.5, 0.0, 0.5)
      0.25: blue (0.0, 0.0, 1.0)
      0.5: green (0.0, 1.0, 0.0)
      0.75: yellow (1.0, 1.0, 0.0)
      1.0: red (1.0, 0.0, 0.0)
    """
    low = 25e6
    high = 35e6
    t = (p - low) / (high - low)
    t = max(0.0, min(1.0, t))
    stops = [
        (0.0, (0.5, 0.0, 0.5)),  # purple
        (0.25, (0.0, 0.0, 1.0)), # blue
        (0.5, (0.0, 1.0, 0.0)),  # green
        (0.75, (1.0, 1.0, 0.0)), # yellow
        (1.0, (1.0, 0.0, 0.0))   # red
    ]
    for i in range(len(stops)-1):
        t0, c0 = stops[i]
        t1, c1 = stops[i+1]
        if t >= t0 and t <= t1:
            factor = (t - t0) / (t1 - t0)
            r = c0[0] + factor * (c1[0] - c0[0])
            g = c0[1] + factor * (c1[1] - c0[1])
            b = c0[2] + factor * (c1[2] - c0[2])
            return (r, g, b)
    return stops[-1][1]

def build_block(xC, yC, thickness):
    """
    Returns 8 vertices for a block (cube-like).
    Bottom face at z=0; top face at z=thickness.
    """
    halfW = cellWidth / 2.0
    halfH = cellHeight / 2.0
    z0 = 0.0
    z1 = thickness
    x0 = xC - halfW
    x1 = xC + halfW
    y0 = yC - halfH
    y1 = yC + halfH
    return [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1)
    ]

block_tris = [
    (0,1,2), (2,3,0),   # bottom face
    (4,5,6), (6,7,4),   # top face
    (0,4,5), (5,1,0),   # side face 1
    (1,5,6), (6,2,1),   # side face 2
    (2,6,7), (7,3,2),   # side face 3
    (3,7,4), (4,0,3)    # side face 4
]

###############################################################################
# 2. Load Simulation JSON
###############################################################################
print("Loading simulation_results.json...")
with open("simulation_results.json", "r") as f:
    sim_data = json.load(f)
print("Simulation data loaded.")

nx = sim_data["nx"]
ny = sim_data["ny"]
dx = sim_data["dx"]
dy = sim_data["dy"]
timeSteps = sim_data["timeSteps"]    # list of times
pressures_flat = sim_data["pressures"]
n_timesteps = len(timeSteps)
print(f"Grid: {nx} x {ny}, total timesteps = {n_timesteps}")

pressures = [np.array(p).reshape((nx, ny)) for p in pressures_flat]
minP = min(np.min(a) for a in pressures)
maxP = max(np.max(a) for a in pressures)
print(f"Global pressure range: [{minP}, {maxP}]")

cellWidth = dx * visualScale
cellHeight = dy * visualScale
pressureScale = cellWidth * 0.4

###############################################################################
# 3. Build Base Mesh (from Timestep 0)
###############################################################################
print("Building base mesh (positions & colors) from first timestep...")
base_positions_list = []
base_colors_list = []
indices_list = []
vertex_count = 0

p0 = pressures[0]
for j in range(ny):
    for i in range(nx):
        xC = (i+0.5)*cellWidth - (nx*cellWidth)/2
        yC = (j+0.5)*cellHeight - (ny*cellHeight)/2
        pVal = p0[i,j]
        frac = (pVal - minP) / max(1e-30, (maxP - minP))
        thickness = base_thickness + frac*pressureScale

        verts = build_block(xC, yC, thickness)
        c = pressure_to_color(pVal)

        for v in verts:
            base_positions_list.extend(v)
            base_colors_list.extend(c)

        for tri in block_tris:
            indices_list.extend([vertex_count + tri[0],
                                 vertex_count + tri[1],
                                 vertex_count + tri[2]])
        vertex_count += 8

base_positions = np.array(base_positions_list, dtype=np.float32).reshape(-1, 3)
base_colors = np.array(base_colors_list, dtype=np.float32).reshape(-1, 3)
indices = np.array(indices_list, dtype=np.uint32)  # we'll convert below if <65536

if vertex_count < 65536:
    indices = indices.astype(np.uint16)
    index_component_type = 5123  # UNSIGNED_SHORT
else:
    index_component_type = 5125  # UNSIGNED_INT

print(f"Base mesh built: {vertex_count} vertices, {len(indices)//3} triangles.")

###############################################################################
# 4. Build Morph Targets (Positions only)
###############################################################################
# We want one morph target for each post-initial timestep.
print(f"Building {n_timesteps-1} morph targets (positions only)...")

base_positions_2D = base_positions  # shape (N,3)
morph_targets_positions = []

for t in range(1, n_timesteps):
    current_positions_list = []
    p_arr = pressures[t]
    for j in range(ny):
        for i in range(nx):
            xC = (i+0.5)*cellWidth - (nx*cellWidth)/2
            yC = (j+0.5)*cellHeight - (ny*cellHeight)/2
            pVal = p_arr[i,j]
            frac = (pVal - minP) / max(1e-30, (maxP - minP))
            thickness = base_thickness + frac*pressureScale

            blockVerts = build_block(xC, yC, thickness)
            current_positions_list.extend(blockVerts)

    current_positions = np.array(current_positions_list, dtype=np.float32).reshape(-1,3)
    if current_positions.shape != base_positions_2D.shape:
        raise ValueError(f"Mismatch in vertex count at timestep {t}!")
    posDelta = current_positions - base_positions_2D
    morph_targets_positions.append(posDelta)

###############################################################################
# 5. Animation: One Channel per Morph Target
###############################################################################
# Because you have ~300 timesteps => 299 morph targets,
# we cannot combine them into one "VEC299" accessor.
# Instead, we create 299 separate channels, each with a SCALAR output.
# For channel i, the array is 0 except at times[i+1], where it's 1.
# We'll do STEP interpolation for abrupt changes.

anim_times = np.array(timeSteps, dtype=np.float32)  # shape: (n_timesteps,)
n_targets = n_timesteps - 1

# Each channel's weight array => length = n_timesteps.  0..0..1..0..0
weight_arrays = []
for i_target in range(n_targets):
    w = np.zeros(n_timesteps, dtype=np.float32)
    # At timeSteps[i_target+1], set weight = 1
    w[i_target+1] = 1.0
    weight_arrays.append(w)

###############################################################################
# 6. Pack All Binary Data
###############################################################################
binary_data = b""
offset = 0

def add_to_buffer(arr_bytes):
    global binary_data, offset
    cur_off = offset
    binary_data += arr_bytes
    offset += len(arr_bytes)
    return (cur_off, len(arr_bytes))

# 6.1 base_positions
bp_bytes = base_positions_2D.astype(np.float32).tobytes()
bp_off, bp_len = add_to_buffer(bp_bytes)

# 6.2 base_colors
bc_bytes = base_colors.astype(np.float32).tobytes()
bc_off, bc_len = add_to_buffer(bc_bytes)

# 6.3 indices
id_bytes = indices.tobytes()
id_off, id_len = add_to_buffer(id_bytes)

# 6.4 morph target positions
morph_pos_info = []
for mt in morph_targets_positions:
    mt_bytes = mt.astype(np.float32).tobytes()
    off_, len_ = add_to_buffer(mt_bytes)
    morph_pos_info.append((off_, len_))

# 6.5 animation times
at_bytes = anim_times.tobytes()
at_off, at_len = add_to_buffer(at_bytes)

# 6.6 each weight array
weight_offsets = []
for w_arr in weight_arrays:
    w_bytes = w_arr.astype(np.float32).tobytes()
    off_, len_ = add_to_buffer(w_bytes)
    weight_offsets.append((off_, len_))

###############################################################################
# 7. Build glTF Scenes/Accessors
###############################################################################
gltf = GLTF2()
gltf.scene = 0
gltf.scenes = [Scene(nodes=[0])]
gltf.buffers = [Buffer(byteLength=len(binary_data))]
gltf.bufferViews = []
gltf.accessors = []

def create_accessor(byteOffset, byteLength, target, componentType, count, type_, min_, max_):
    bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=byteOffset,
        byteLength=byteLength,
        target=target
    ))
    acc_idx = len(gltf.accessors)
    acc_kwargs = dict(
        bufferView=bv_idx,
        byteOffset=0,
        componentType=componentType,
        count=count,
        type=type_
    )
    if min_ is not None: acc_kwargs["min"] = min_
    if max_ is not None: acc_kwargs["max"] = max_
    gltf.accessors.append(Accessor(**acc_kwargs))
    return acc_idx

N = base_positions_2D.shape[0]  # total vertices
# 7.1 base_positions accessor
pos_min = base_positions_2D.min(axis=0).tolist()
pos_max = base_positions_2D.max(axis=0).tolist()
bp_acc = create_accessor(
    bp_off, bp_len,
    34962,  # ARRAY_BUFFER
    5126,   # FLOAT
    N,
    "VEC3",
    pos_min,
    pos_max
)

# 7.2 base_colors accessor
col_min = base_colors.min(axis=0).tolist()
col_max = base_colors.max(axis=0).tolist()
bc_acc = create_accessor(
    bc_off, bc_len,
    34962,
    5126,
    N,
    "VEC3",
    col_min,
    col_max
)

# 7.3 indices accessor
indices_count = len(indices)
id_acc = create_accessor(
    id_off, id_len,
    34963,  # ELEMENT_ARRAY_BUFFER
    index_component_type,
    indices_count,
    "SCALAR",
    None,
    None
)

# 7.4 morph target position accessors
morph_pos_accessors = []
for i, (off_, len_) in enumerate(morph_pos_info):
    mt = morph_targets_positions[i]
    mt_min = mt.min(axis=0).tolist()
    mt_max = mt.max(axis=0).tolist()
    acc_idx = create_accessor(
        off_, len_,
        34962,
        5126,
        N,
        "VEC3",
        mt_min,
        mt_max
    )
    morph_pos_accessors.append(acc_idx)

# 7.5 animation times accessor
t_min = float(anim_times.min())
t_max = float(anim_times.max())
times_acc = create_accessor(
    at_off, at_len,
    None,  # no buffer target
    5126,  # FLOAT
    n_timesteps,
    "SCALAR",
    [t_min],
    [t_max]
)

# 7.6 weight accessors
# each is length n_timesteps, SCALAR
weight_accessors = []
for i, (off_, len_) in enumerate(weight_offsets):
    w_arr = weight_arrays[i]
    w_min = float(w_arr.min())
    w_max = float(w_arr.max())
    wa_idx = create_accessor(
        off_, len_,
        None,
        5126,  # FLOAT
        n_timesteps,
        "SCALAR",
        [w_min],
        [w_max]
    )
    weight_accessors.append(wa_idx)

###############################################################################
# 8. Create the Mesh with Morph Targets
###############################################################################
# glTF doesn't allow "COLOR_0" in morph targets, but we do store it in the base.
# The 'targets' array includes only "POSITION" for each morph target.
targets_list = []
for i_pos in morph_pos_accessors:
    targets_list.append({"POSITION": i_pos})

primitive = Primitive(
    attributes={
        "POSITION": bp_acc,
        "COLOR_0": bc_acc
    },
    indices=id_acc,
    targets=targets_list
)
mesh = Mesh(primitives=[primitive])
gltf.meshes = [mesh]

node = Node(mesh=0, name="ReservoirMesh")
gltf.nodes = [node]

###############################################################################
# 9. Create an Animation with One Channel per Morph Target
###############################################################################
# - We have n_targets morph targets
# - Each morph target will be controlled by a separate channel.
# - Each channel uses the same times_acc, but a different weight_accessors[i].
# - The output is SCALAR of length = n_timesteps (with 0 except 1 at step i+1).
# - Use 'STEP' for abrupt changes.

anim_samplers = []
anim_channels = []
for i_target in range(n_targets):
    s_idx = len(anim_samplers)
    sampler = AnimationSampler(
        input=times_acc,
        output=weight_accessors[i_target],
        interpolation="STEP"
    )
    anim_samplers.append(sampler)
    c_idx = len(anim_channels)
    channel = AnimationChannel(
        sampler=s_idx,
        target={
            "node": 0,
            "path": "weights",
            # Nonstandard but helps some viewers interpret morph index:
            "extras": {"_targetIndex": i_target}
        }
    )
    anim_channels.append(channel)

animation = Animation(
    samplers=anim_samplers,
    channels=anim_channels,
    name="MultiChannelAnimation"
)
gltf.animations = [animation]

###############################################################################
# 10. Finalize and Save
###############################################################################
print("Embedding binary data...")
b64_data = base64.b64encode(binary_data).decode("utf-8")
gltf.buffers[0].uri = "data:application/octet-stream;base64," + b64_data
gltf.convert_buffers(BufferFormat.DATAURI)

out_name = "simulation_animation.gltf"
gltf.save(out_name)
print(f"Saved glTF file: {out_name}")
print("Done.")
