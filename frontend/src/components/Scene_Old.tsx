// components/Scene.tsx
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { DoubleSide } from 'three';

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
  pressures: number[][];
  timeSteps: number[];
  fractures?: FractureData[];
}

export interface HoverInfo {
  i: number;
  j: number;
  pressure: number;
}

interface ReservoirBlocksProps {
  simulationData: SimulationData;
  currentTimeIndex: number;
  gridOpacity: number;
  onHover?: (hoverInfo: HoverInfo | null) => void;
}

const pressureToColor = (pressure: number, minP: number, maxP: number): string => {
  const fraction = (pressure - minP) / (maxP - minP);
  const clamped = Math.max(0, Math.min(fraction, 1));
  const hue = 240 - 240 * clamped;
  return `hsl(${hue}, 100%, 70%)`;
};

const ReservoirBlocks: React.FC<ReservoirBlocksProps> = ({
  simulationData,
  currentTimeIndex,
  gridOpacity,
  onHover,
}) => {
  const cells = useMemo(() => {
    if (
      !simulationData.nx ||
      !simulationData.ny ||
      !simulationData.dx ||
      !simulationData.dy ||
      !simulationData.pressures
    ) {
      return [];
    }
    const { nx, ny, dx, dy, pressures } = simulationData;
    const visualScale = 0.2;
    const cellWidth = Number(dx) * visualScale;
    const cellHeight = Number(dy) * visualScale;
    const pFlat = pressures[currentTimeIndex];
    const minP = Math.min(...pFlat);
    const maxP = Math.max(...pFlat);
    const cellArray: { i: number; j: number; color: string; pressure: number }[] = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const index = i + j * nx;
        const p = pFlat[index];
        const color = pressureToColor(p, minP, maxP);
        cellArray.push({ i, j, color, pressure: p });
      }
    }
    return cellArray;
  }, [simulationData, currentTimeIndex]);

  if (
    !simulationData.nx ||
    !simulationData.ny ||
    !simulationData.dx ||
    !simulationData.dy ||
    !simulationData.pressures
  ) {
    return <group />;
  }
  const { nx, ny, dx, dy } = simulationData;
  const visualScale = 0.2;
  const cellWidth = Number(dx) * visualScale;
  const cellHeight = Number(dy) * visualScale;
  const offsetX = (nx * cellWidth) / 2;
  const offsetY = (ny * cellHeight) / 2;

  return (
    <group>
      {cells.map(({ i, j, color, pressure }, index) => (
        <mesh
          key={index}
          position={[i * cellWidth - offsetX, j * cellHeight - offsetY, 0]}
          onPointerOver={(e) => {
            e.stopPropagation();
            onHover && onHover({ i, j, pressure });
          }}
          onPointerOut={(e) => {
            e.stopPropagation();
            onHover && onHover(null);
          }}
        >
          <boxGeometry args={[cellWidth, cellHeight, cellWidth * 0.5]} />
          <meshStandardMaterial
            color={color}
            opacity={gridOpacity}
            transparent={gridOpacity < 1}
          />
        </mesh>
      ))}
    </group>
  );
};

interface WellLineProps {
  simulationData: SimulationData;
}

const WellLine: React.FC<WellLineProps> = ({ simulationData }) => {
  if (!simulationData.nx || !simulationData.ny || !simulationData.dx || !simulationData.dy) {
    return <group />;
  }
  const { nx, ny, dx, dy } = simulationData;
  const visualScale = 0.2;
  const cellWidth = Number(dx) * visualScale;
  const cellHeight = Number(dy) * visualScale;
  const midY = Math.floor(ny / 2);
  const offsetX = (nx * cellWidth) / 2;
  const offsetY = (ny * cellHeight) / 2;
  const startX = 0 - offsetX;
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
        <bufferAttribute attach="attributes-position" array={vertices} count={points.length} itemSize={3} />
      </bufferGeometry>
      <lineBasicMaterial attach="material" color="black" linewidth={4} />
    </line>
  );
};

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
    return <group />;
  }
  const { fractures, dx, dy, nx, ny } = simulationData;
  const visualScale = 0.2;
  const cellWidth = Number(dx) * visualScale;
  const cellHeight = Number(dy) * visualScale;
  const fractureThickness = cellWidth * 0.1;
  const midY = Math.floor(ny / 2);
  const offsetX = (nx * cellWidth) / 2;
  const offsetY = (ny * cellHeight) / 2;

  return (
    <>
      {fractures.map((f, idx) => {
        // If alternative properties exist, you can adjust the fallback here
        const x_start = Number(f.x_start);
        const centerY = midY * cellHeight - offsetY;
        const fractureLength = Number(f.length_cells) * cellHeight;
        const centerX = x_start * cellWidth - offsetX;
        const centerZ = cellWidth * 0.3; // raised slightly above the grid
        return (
          <mesh key={idx} position={[centerX, centerY, centerZ]}>
            <planeGeometry args={[fractureThickness, fractureLength]} />
            <meshStandardMaterial color="yellow" side={DoubleSide} transparent opacity={0.8} />
          </mesh>
        );
      })}
    </>
  );
};

interface SceneProps {
  simulationData: SimulationData;
  currentTimeIndex?: number;
  gridOpacity?: number;
  onHover?: (hoverInfo: HoverInfo | null) => void;
}

const Scene: React.FC<SceneProps> = ({
  simulationData,
  currentTimeIndex = 0,
  gridOpacity = 1,
  onHover,
}) => {
  if (!simulationData) return <group />;
  return (
    <Canvas style={{ width: '100%', height: '100%' }}>
      <OrbitControls />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} />
      <ReservoirBlocks
        simulationData={simulationData}
        currentTimeIndex={currentTimeIndex}
        gridOpacity={gridOpacity}
        onHover={onHover}
      />
      <WellLine simulationData={simulationData} />
      <FracturePlanes simulationData={simulationData} />
    </Canvas>
  );
};

export default Scene;
