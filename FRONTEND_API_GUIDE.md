# GitProbe API Integration Guide

**Version**: 1.0  
**Base URL**: `http://localhost:8000/api/v1`  
**Documentation**: `http://localhost:8000/docs`

## Overview

GitProbe is a GitHub repository analysis API that extracts function call graphs from Python codebases. It provides detailed insights into code structure, function relationships, and enables interactive exploration of repository architectures.

### Key Features
- 🔍 **Repository Analysis**: Clone and analyze GitHub repositories
- 📊 **Function Extraction**: Identify all functions with metadata
- 🔗 **Relationship Mapping**: Track function call relationships  
- 🎯 **Selective Analysis**: Filter and select specific functions
- 📤 **Multiple Export Formats**: JSON, SVG, Cytoscape.js
- 🏷️ **Custom Naming**: Rename functions for better UX

---

## Quick Start

### 1. Analyze a Repository
```javascript
const response = await fetch('/api/v1/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    repo_url: 'https://github.com/psf/requests',
    include_patterns: '*.py',
    exclude_patterns: 'tests/'
  })
});
const analysis = await response.json();
```

### 2. Get Analysis Summary
```javascript
const summary = await fetch(`/api/v1/analysis/${analysisId}/summary`).then(r => r.json());
console.log(`Found ${summary.summary.total_functions} functions`);
```

### 3. Export for Visualization
```javascript
const graph = await fetch(`/api/v1/analysis/${analysisId}/export/cytoscape`).then(r => r.json());
// Use with Cytoscape.js, D3.js, or other graph libraries
```

---

## API Endpoints

### Analysis Operations

#### `POST /analyze`
Analyze a GitHub repository and extract function call graph.

**Parameters:**
- `repo_url` (required): GitHub repository URL
- `include_patterns` (optional): File patterns to include (e.g., `*.py`, `src/`)
- `exclude_patterns` (optional): File patterns to exclude (e.g., `tests/`, `__pycache__/`)

**Response:**
```typescript
interface AnalysisResult {
  repository: {
    url: string;
    name: string;
    clone_path: string;
    analysis_id: string;
  };
  functions: Function[];
  relationships: CallRelationship[];
  file_tree: FileTree;
  summary: {
    total_functions: number;
    total_relationships: number;
    resolved_relationships: number;
    files_analyzed: number;
    languages: string[];
  };
}
```

**Example:**
```bash
curl -X POST "/api/v1/analyze?repo_url=https://github.com/psf/requests&include_patterns=*.py&exclude_patterns=tests/"
```

#### `GET /analysis/{analysis_id}`
Retrieve complete analysis results.

#### `GET /analysis/{analysis_id}/summary`
Get analysis overview with metrics.

**Response:**
```typescript
interface Summary {
  repository: Repository;
  summary: {
    total_functions: number;
    total_relationships: number;
    resolved_relationships: number;
    files_analyzed: number;
    languages: string[];
  };
  function_count: number;
  relationship_count: number;
  files: string[];
  classes: string[];
}
```

#### `GET /analysis/{analysis_id}/file-tree`
Get repository file structure.

**Response:**
```typescript
interface FileTree {
  type: 'directory' | 'file';
  name: string;
  path: string;
  children?: FileTree[];
  size_kb?: number;
  estimated_tokens?: number;
  extension?: string;
}
```

### Function Operations

#### `GET /analysis/{analysis_id}/functions`
List all functions in the analysis.

**Response:**
```typescript
interface Function {
  name: string;
  file_path: string;
  line_start: number;
  line_end?: number;
  parameters?: string[];
  docstring?: string;
  is_method: boolean;
  class_name?: string;
  code_snippet?: string;
  display_name?: string;
}
```

#### `GET /analysis/{analysis_id}/functions/{node_id}`
Get detailed information about a specific function.

**Node ID Format**: `{file_path}:{function_name}`  
**Example**: `src/requests/api.py:get`

**Response:**
```typescript
interface FunctionDetails {
  node_id: string;
  original_name: string;
  display_name: string;
  file_path: string;
  line_start: number;
  line_end?: number;
  parameters?: string[];
  docstring?: string;
  is_method: boolean;
  class_name?: string;
  code_snippet?: string;
  calls: {
    incoming: number;
    outgoing: number;
    incoming_details: Array<{
      caller: string;
      line?: number;
    }>;
    outgoing_details: Array<{
      callee: string;
      line?: number;
    }>;
  };
}
```

#### `GET /analysis/{analysis_id}/functions/{node_id}/code`
Get copy-friendly function code (plain text).

**Response**: Raw function code as plain text

#### `PUT /analysis/{analysis_id}/rename/{node_id}`
Rename a function for display purposes.

**Request Body**: Raw string with new name

**Response:**
```typescript
interface RenameResponse {
  node_id: string;
  new_name: string;
  success: boolean;
}
```

### Selection Operations

#### `POST /analysis/{analysis_id}/select`
Select multiple functions for operations.

**Request:**
```typescript
interface SelectRequest {
  // Array of node IDs
  node_ids: string[];
}
```

**Response:**
```typescript
interface SelectResponse {
  analysis_id: string;
  selected_nodes: string[];
  count: number;
}
```

**Example:**
```javascript
await fetch(`/api/v1/analysis/${analysisId}/select`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify([
    'src/requests/api.py:get',
    'src/requests/api.py:post'
  ])
});
```

### Export Operations

#### `GET /analysis/{analysis_id}/export/json`
Export analysis data as JSON.

**Query Parameters:**
- `selected_only` (boolean): Export only selected functions
- `include_code` (boolean): Include function code snippets
- `include_relationships` (boolean): Include call relationships

**Response:**
```typescript
interface ExportData {
  analysis: AnalysisResult;
  selection?: NodeSelection;
  export_type: string;
  generated_at: string;
}
```

#### `GET /analysis/{analysis_id}/export/svg`
Export call graph as SVG diagram.

**Response**: SVG content (Content-Type: image/svg+xml)

#### `GET /analysis/{analysis_id}/export/cytoscape`
Export as Cytoscape.js format for interactive graphs.

**Response:**
```typescript
interface CytoscapeData {
  elements: Array<{
    data: {
      id: string;
      label?: string;
      source?: string;
      target?: string;
      file?: string;
      type?: string;
      class?: string;
      line?: number;
    };
    classes?: string;
  }>;
  style: Array<{
    selector: string;
    style: Record<string, any>;
  }>;
}
```

---

## Frontend Integration Examples

### React Hook for Repository Analysis

```typescript
import { useState, useCallback } from 'react';

interface UseAnalysisReturn {
  analysis: AnalysisResult | null;
  loading: boolean;
  error: string | null;
  analyzeRepo: (url: string, filters?: AnalysisFilters) => Promise<void>;
}

interface AnalysisFilters {
  includePatterns?: string[];
  excludePatterns?: string[];
}

export const useAnalysis = (): UseAnalysisReturn => {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeRepo = useCallback(async (url: string, filters?: AnalysisFilters) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({ repo_url: url });
      
      filters?.includePatterns?.forEach(pattern => 
        params.append('include_patterns', pattern)
      );
      filters?.excludePatterns?.forEach(pattern => 
        params.append('exclude_patterns', pattern)
      );

      const response = await fetch(`/api/v1/analyze?${params}`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`);
      
      const result = await response.json();
      setAnalysis(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, []);

  return { analysis, loading, error, analyzeRepo };
};
```

### Vue.js Component Example

```vue
<template>
  <div class="gitprobe-integration">
    <form @submit.prevent="analyzeRepository">
      <input 
        v-model="repoUrl" 
        placeholder="https://github.com/user/repo" 
        required 
      />
      <button type="submit" :disabled="loading">
        {{ loading ? 'Analyzing...' : 'Analyze Repository' }}
      </button>
    </form>
    
    <div v-if="analysis" class="results">
      <h3>{{ analysis.repository.name }}</h3>
      <p>Functions: {{ analysis.summary.total_functions }}</p>
      <p>Relationships: {{ analysis.summary.total_relationships }}</p>
      
      <div class="functions-grid">
        <div 
          v-for="func in analysis.functions" 
          :key="`${func.file_path}:${func.name}`"
          class="function-card"
          @click="selectFunction(func)"
        >
          <h4>{{ func.display_name || func.name }}</h4>
          <p>{{ func.file_path }}:{{ func.line_start }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const repoUrl = ref('');
const loading = ref(false);
const analysis = ref<AnalysisResult | null>(null);
const selectedFunctions = ref<string[]>([]);

const analyzeRepository = async () => {
  loading.value = true;
  try {
    const response = await fetch('/api/v1/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ repo_url: repoUrl.value })
    });
    analysis.value = await response.json();
  } finally {
    loading.value = false;
  }
};

const selectFunction = async (func: Function) => {
  const nodeId = `${func.file_path}:${func.name}`;
  selectedFunctions.value.push(nodeId);
  
  // Update selection on server
  await fetch(`/api/v1/analysis/${analysis.value?.repository.analysis_id}/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(selectedFunctions.value)
  });
};
</script>
```

### Cytoscape.js Graph Visualization

```typescript
import cytoscape from 'cytoscape';

export const createCallGraphVisualization = async (
  analysisId: string, 
  container: HTMLElement
) => {
  // Fetch graph data
  const graphData = await fetch(`/api/v1/analysis/${analysisId}/export/cytoscape`)
    .then(response => response.json());

  // Initialize Cytoscape
  const cy = cytoscape({
    container,
    elements: graphData.elements,
    style: [
      ...graphData.style,
      {
        selector: 'node:selected',
        style: {
          'border-color': '#ff6b6b',
          'border-width': 3
        }
      }
    ],
    layout: {
      name: 'cose',
      animate: true,
      animationDuration: 1000
    }
  });

  // Handle node clicks
  cy.on('tap', 'node', async (event) => {
    const nodeId = event.target.id();
    
    // Fetch detailed function information
    const details = await fetch(`/api/v1/analysis/${analysisId}/functions/${nodeId}`)
      .then(r => r.json());
    
    // Show function details in sidebar/modal
    showFunctionDetails(details);
  });

  return cy;
};
```

---

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Resource not found (analysis/function not found)
- `500`: Internal Server Error (analysis failed)

### Error Response Format
```typescript
interface ErrorResponse {
  detail: string;
}
```

### Common Errors
- **"Analysis not found"**: Invalid analysis ID
- **"Function not found"**: Invalid node ID format
- **"Repository URL is required"**: Missing repo_url parameter
- **"Analysis failed: ..."**: Repository cloning or analysis error

---

## Best Practices

### Performance
- **Cache analysis results** in your frontend state management
- **Use analysis IDs** to avoid re-analyzing the same repository
- **Implement pagination** for large function lists
- **Debounce search/filter inputs**

### UX Recommendations
- **Show progress indicators** during analysis (can take 10-30 seconds)
- **Display file tree** for repository exploration
- **Implement function search/filtering**
- **Provide copy-to-clipboard** for function code
- **Allow custom function renaming** for better user experience

### Data Management
```typescript
// Example state structure
interface AppState {
  analyses: Record<string, AnalysisResult>;
  currentAnalysis: string | null;
  selectedFunctions: string[];
  customNames: Record<string, string>;
  filters: {
    searchTerm: string;
    fileFilter: string;
    showMethodsOnly: boolean;
  };
}
```

### Integration Workflow
1. **Repository Input** → User provides GitHub URL
2. **Analysis Request** → POST to `/analyze` with filters
3. **Progress Indication** → Show loading state (10-30s)
4. **Results Display** → Show summary, file tree, function list
5. **Interactive Exploration** → Click functions for details
6. **Selection & Export** → Multi-select and export functionality
7. **Visualization** → Render call graphs with Cytoscape.js/D3.js

---

## TypeScript Definitions

```typescript
// Complete type definitions for GitProbe API
export interface Repository {
  url: string;
  name: string;
  clone_path: string;
  analysis_id: string;
}

export interface Function {
  name: string;
  file_path: string;
  line_start: number;
  line_end?: number;
  parameters?: string[];
  docstring?: string;
  is_method: boolean;
  class_name?: string;
  code_snippet?: string;
  display_name?: string;
}

export interface CallRelationship {
  caller: string;
  callee: string;
  call_line?: number;
  is_resolved: boolean;
}

export interface AnalysisResult {
  repository: Repository;
  functions: Function[];
  relationships: CallRelationship[];
  file_tree: Record<string, any>;
  summary: {
    total_functions: number;
    total_relationships: number;
    resolved_relationships: number;
    files_analyzed: number;
    languages: string[];
  };
}

export interface NodeSelection {
  selected_nodes: string[];
  include_relationships: boolean;
  custom_names: Record<string, string>;
}

export interface ExportData {
  analysis: AnalysisResult;
  selection?: NodeSelection;
  export_type: string;
  generated_at: string;
}
```

---

## Support

- **API Documentation**: `http://localhost:8000/docs`
- **Interactive Testing**: Use FastAPI's built-in Swagger UI
- **Analysis ID Format**: URL-safe string (use returned `analysis_id` from repository object)
- **Node ID Format**: `{file_path}:{function_name}` (URL-encode if needed)

For questions or issues, refer to the interactive API documentation or test endpoints directly in the Swagger UI. 