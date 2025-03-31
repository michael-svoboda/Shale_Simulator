// components/Scene.tsx
import React, { useMemo, useEffect, useRef, useCallback } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils';

// --- Interfaces ---
export interface FractureData {
  x_start: number;
  y_start: number;
  length_cells: number;
}

export interface SimulationData {
  nx: number;
  ny: number;
  dx: number;
  dy: number;
  pressures: number[][]; // shape: [timeStep][nx*ny]
  timeSteps: number[];
  fractures?: FractureData[];
}

export interface HoverInfo {
  i: number;
  j: number;
  pressure: number;
}

export interface SceneProps {
  simulationData: SimulationData;
  currentTimeIndex?: number;
  gridOpacity?: number;
  onHover?: (hoverInfo: HoverInfo | null) => void;
  cameraDistance?: number;
  // New props for 3D mode and height exaggeration.
  is3D?: boolean;
  heightExaggeration?: number;
  restrictVertical?: boolean;
}

// --- MergedReservoirBlocks Component ---
// (This component creates the merged block geometry for the reservoir.)
interface MergedReservoirBlocksProps {
  simulationData: SimulationData;
  currentTimeIndex: number;
  gridOpacity: number;
  onHover?: (hoverInfo: HoverInfo | null) => void;
  is3D: boolean;
  heightExaggeration: number;
}

function pressureToColor(
  pressure: number,
  minP: number,
  maxP: number
): THREE.Color {
  const fraction = (pressure - minP) / (maxP - minP || 1);
  const clamped = Math.max(0, Math.min(1, fraction));
  const hue = 240 - 240 * clamped; // 240=blue -> 0=red
  const color = new THREE.Color();
  color.setHSL(hue / 360, 1, 0.6);
  return color;
}

const MergedReservoirBlocks: React.FC<MergedReservoirBlocksProps> = ({
  simulationData,
  currentTimeIndex,
  gridOpacity,
  onHover,
  is3D,
  heightExaggeration,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  const { mergedGeometry, faceToCellIndex } = useMemo(() => {
    if (!simulationData) {
      return { mergedGeometry: new THREE.BufferGeometry(), faceToCellIndex: [] };
    }
    const { nx, ny, dx, dy, pressures } = simulationData;
    if (!nx || !ny || !dx || !dy || !pressures?.length) {
      return { mergedGeometry: new THREE.BufferGeometry(), faceToCellIndex: [] };
    }
    const pFlat = pressures[currentTimeIndex];
    const minP = Math.min(...pFlat);
    const maxP = Math.max(...pFlat);
    const visualScale = 0.2;
    const cellWidth = dx * visualScale;
    const cellHeight = dy * visualScale;
    const halfWidth = (nx * cellWidth) / 2;
    const halfHeight = (ny * cellHeight) / 2;

    // Define base thickness and extra thickness based on pressure.
    const baseThickness = cellWidth * 0.1;
    const pressureScale = cellWidth * 0.4;

    const geometries: THREE.BufferGeometry[] = [];
    const faceToCellIndex: { i: number; j: number; pressure: number }[] = [];
    let faceIndexOffset = 0;

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = i + j * nx;
        const pressureVal = pFlat[idx];
        const color = pressureToColor(pressureVal, minP, maxP);
        const thickness = is3D
          ? baseThickness + heightExaggeration * ((pressureVal - minP) / (maxP - minP || 1)) * pressureScale
          : baseThickness;
        const boxGeom = new THREE.BoxGeometry(cellWidth, cellHeight, thickness);
        const xPos = i * cellWidth - halfWidth + cellWidth / 2;
        const yPos = j * cellHeight - halfHeight + cellHeight / 2;
        const zPos = thickness / 2;
        boxGeom.translate(xPos, yPos, zPos);

        const positions = boxGeom.getAttribute('position');
        const vertexCount = positions.count;
        const colors = new Float32Array(vertexCount * 3);
        for (let v = 0; v < vertexCount; v++) {
          colors[v * 3] = color.r;
          colors[v * 3 + 1] = color.g;
          colors[v * 3 + 2] = color.b;
        }
        boxGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        for (let faceIdx = 0; faceIdx < 12; faceIdx++) {
          faceToCellIndex[faceIndexOffset + faceIdx] = { i, j, pressure: pressureVal };
        }
        faceIndexOffset += 12;
        geometries.push(boxGeom);
      }
    }
    const merged = mergeGeometries(geometries, false);
    return { mergedGeometry: merged, faceToCellIndex };
  }, [simulationData, currentTimeIndex, is3D, heightExaggeration]);

  const material = useMemo(() => {
    if (is3D) {
      return new THREE.MeshStandardMaterial({
        vertexColors: true,
        transparent: gridOpacity < 1,
        opacity: gridOpacity,
        metalness: 0.2,
        roughness: 0.5,
      });
    }
    return new THREE.MeshStandardMaterial({
      vertexColors: true,
      transparent: gridOpacity < 1,
      opacity: gridOpacity,
    });
  }, [gridOpacity, is3D]);

  const handlePointerMove = useCallback(
    (e: any) => {
      if (!onHover) return;
      e.stopPropagation();
      const fIdx = e.faceIndex;
      if (fIdx == null) {
        onHover(null);
        return;
      }
      const cellInfo = faceToCellIndex[fIdx];
      if (!cellInfo) {
        onHover(null);
        return;
      }
      onHover({ i: cellInfo.i, j: cellInfo.j, pressure: cellInfo.pressure });
    },
    [faceToCellIndex, onHover]
  );

  const handlePointerOut = useCallback(() => {
    onHover && onHover(null);
  }, [onHover]);

  if (!mergedGeometry || mergedGeometry.attributes.position.count === 0) {
    return null;
  }

  return (
    <mesh
      ref={meshRef}
      geometry={mergedGeometry}
      material={material}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    />
  );
};

// --- WellLine Component ---
interface WellLineProps {
  simulationData: SimulationData;
}
const WellLine: React.FC<WellLineProps> = ({ simulationData }) => {
  if (!simulationData.nx || !simulationData.ny || !simulationData.dx || !simulationData.dy) {
    return null;
  }
  const { nx, ny, dx, dy } = simulationData;
  const visualScale = 0.2;
  const cellWidth = Number(dx) * visualScale;
  const cellHeight = Number(dy) * visualScale;
  const midY = Math.floor(ny / 2);
  const offsetX = (nx * cellWidth) / 2;
  const offsetY = (ny * cellHeight) / 2;
  const startX = -offsetX;
  const endX = nx * cellWidth - offsetX;
  const y = midY * cellHeight - offsetY;
  const z = 0.6;
  const points = [
    [startX, y, z],
    [endX, y, z],
  ];
  const vertices = new Float32Array(points.flat());
  return (
    <line>
      <bufferGeometry attach="geometry">
        <bufferAttribute attach="attributes-position" args={[vertices, 3]} />
      </bufferGeometry>
      <lineBasicMaterial attach="material" color="black" linewidth={4} />
    </line>
  );
};

// --- FracturePlanes Component ---
interface FracturePlanesProps {
  simulationData: SimulationData;
}
const FracturePlanes: React.FC<FracturePlanesProps> = ({ simulationData }) => {
  if (
    !simulationData.nx ||
    !simulationData.ny ||
    !simulationData.dx ||
    !simulationData.dy ||
    !simulationData.fractures
  ) {
    return null;
  }
  const { fractures, dx, dy, nx, ny } = simulationData;
  const visualScale = 0.2;
  const cellWidth = Number(dx) * visualScale;
  const cellHeight = Number(dy) * visualScale;
  const midY = Math.floor(ny / 2);
  const offsetX = (nx * cellWidth) / 2;
  const offsetY = (ny * cellHeight) / 2;
  return (
    <>
      {fractures.map((f, idx) => {
        const x_start = Number(f.x_start);
        const centerY = midY * cellHeight - offsetY;
        const fractureLength = Number(f.length_cells) * cellHeight;
        const centerX = x_start * cellWidth - offsetX + cellWidth / 2;
        const centerZ = cellWidth * 0.3;
        return (
          <mesh key={idx} position={[centerX, centerY, centerZ]}>
            <planeGeometry args={[cellWidth * 0.1, fractureLength]} />
            <meshStandardMaterial
              color="yellow"
              side={THREE.DoubleSide}
              transparent
              opacity={0.8}
            />
          </mesh>
        );
      })}
    </>
  );
};

// --- SceneContents Component ---
const SceneContents: React.FC<SceneProps> = ({
  simulationData,
  currentTimeIndex = 0,
  gridOpacity = 1,
  onHover,
  cameraDistance = 10,
  is3D = true,
  heightExaggeration = 1,
  restrictVertical = false,
}) => {
  const { camera, gl } = useThree();
  const orbitControlsRef = useRef<any>();

  useEffect(() => {
    camera.position.set(0, 0, cameraDistance);
  }, [cameraDistance, camera]);

  useEffect(() => {
    if (is3D) {
      gl.shadowMap.enabled = true;
      gl.shadowMap.type = THREE.PCFSoftShadowMap;
    }
  }, [gl, is3D]);

  // Hotkey: "c" to center camera target
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'c' && orbitControlsRef.current) {
        orbitControlsRef.current.target.set(0, 0, 0);
        orbitControlsRef.current.update();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      <OrbitControls
        ref={orbitControlsRef}
        enableZoom={false}
        minPolarAngle={restrictVertical ? Math.PI / 2 : 0}
        maxPolarAngle={restrictVertical ? Math.PI / 2 : Math.PI}
      />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={0.4} />
      {is3D && (
        <>
          <hemisphereLight skyColor={0xffffff} groundColor={0x444444} intensity={0.5} />
          <directionalLight
            position={[10, 10, 10]}
            intensity={0.8}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
            shadow-camera-near={0.5}
            shadow-camera-far={500}
          />
        </>
      )}
      <MergedReservoirBlocks
        simulationData={simulationData}
        currentTimeIndex={currentTimeIndex!}
        gridOpacity={gridOpacity!}
        onHover={onHover}
        is3D={is3D}
        heightExaggeration={heightExaggeration}
      />
      <WellLine simulationData={simulationData} />
      <FracturePlanes simulationData={simulationData} />
    </>
  );
};

/* ----------------------------------------------------------
   Scene: Top-level component wrapping the Canvas.
-----------------------------------------------------------*/
const Scene: React.FC<SceneProps> = (props) => {
  const { simulationData, cameraDistance = 10 } = props;
  if (!simulationData) return null;
  return (
    <Canvas
      style={{ width: '100%', height: '100%' }}
      camera={{ position: [0, 0, cameraDistance], fov: 75 }}
    >
      <SceneContents {...props} />
    </Canvas>
  );
};

export default Scene;
