// TruthLens API Integration Library
// This file provides functions to interact with the TruthLens backend

import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_TRUTHLENS_API_URL || 'http://localhost:8000';

export interface TruthLensConfig {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
}

export interface FactCheckRequest {
  input_type: 'text' | 'url' | 'file';
  content: string;
  options?: {
    language?: string;
    include_evidence?: boolean;
    include_manipulation_detection?: boolean;
    max_evidence_count?: number;
  };
}

export interface FactCheckResponse {
  claims: Claim[];
  evidence: Evidence[];
  manipulation_detected: boolean;
  manipulation_types: string[];
  overall_verdict: string;
  confidence: number;
  processing_time: number;
  metadata: {
    model_version: string;
    timestamp: string;
    input_length: number;
  };
}

export interface Claim {
  id: string;
  text: string;
  subject: string;
  predicate: string;
  object: string;
  checkworthiness: number;
  context: {
    negation: boolean;
    modality: string;
    conditional_trigger: string;
    sarcasm_score: number;
    attribution: string;
  };
}

export interface Evidence {
  id: string;
  title: string;
  content: string;
  url: string;
  source_type: string;
  relevance_score: number;
  stance: string;
  confidence: number;
  published_date?: string;
  author?: string;
}

class TruthLensAPI {
  private config: TruthLensConfig;
  private client: any;

  constructor(config: TruthLensConfig = {}) {
    this.config = {
      baseUrl: API_BASE_URL,
      timeout: 30000,
      ...config
    };

    this.client = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` })
      }
    });
  }

  /**
   * Perform fact-checking on input content
   */
  async factCheck(request: FactCheckRequest): Promise<FactCheckResponse> {
    try {
      const response = await this.client.post('/api/v1/fact-check', request);
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(`TruthLens API Error: ${error.response.data?.message || error.response.statusText}`);
      } else if (error.request) {
        throw new Error('Network error: Unable to connect to TruthLens API');
      } else {
        throw new Error(`Request error: ${error.message}`);
      }
    }
  }

  /**
   * Extract claims from text without full fact-checking
   */
  async extractClaims(text: string): Promise<Claim[]> {
    try {
      const response = await this.client.post('/api/v1/extract-claims', { text });
      return response.data.claims;
    } catch (error: any) {
      throw new Error(`Claim extraction error: ${error.message}`);
    }
  }

  /**
   * Search for evidence related to a claim
   */
  async searchEvidence(query: string, options?: {
    max_results?: number;
    source_types?: string[];
    date_range?: string;
  }): Promise<Evidence[]> {
    try {
      const response = await this.client.post('/api/v1/search-evidence', {
        query,
        ...options
      });
      return response.data.evidence;
    } catch (error: any) {
      throw new Error(`Evidence search error: ${error.message}`);
    }
  }

  /**
   * Check API health and status
   */
  async healthCheck(): Promise<{
    status: string;
    version: string;
    models_loaded: boolean;
    uptime: number;
  }> {
    try {
      const response = await this.client.get('/api/v1/health');
      return response.data;
    } catch (error: any) {
      throw new Error(`Health check error: ${error.message}`);
    }
  }

  /**
   * Get supported languages
   */
  async getSupportedLanguages(): Promise<string[]> {
    try {
      const response = await this.client.get('/api/v1/languages');
      return response.data.languages;
    } catch (error: any) {
      throw new Error(`Language fetch error: ${error.message}`);
    }
  }

  /**
   * Get trusted sources list
   */
  async getTrustedSources(): Promise<{
    government: string[];
    news: string[];
    academic: string[];
    fact_checkers: string[];
  }> {
    try {
      const response = await this.client.get('/api/v1/trusted-sources');
      return response.data;
    } catch (error: any) {
      throw new Error(`Trusted sources fetch error: ${error.message}`);
    }
  }
}

// Create default instance
export const truthLensAPI = new TruthLensAPI();

// Export individual functions for convenience
export const factCheck = (request: FactCheckRequest) => truthLensAPI.factCheck(request);
export const extractClaims = (text: string) => truthLensAPI.extractClaims(text);
export const searchEvidence = (query: string, options?: any) => truthLensAPI.searchEvidence(query, options);
export const healthCheck = () => truthLensAPI.healthCheck();
export const getSupportedLanguages = () => truthLensAPI.getSupportedLanguages();
export const getTrustedSources = () => truthLensAPI.getTrustedSources();

export default TruthLensAPI;
