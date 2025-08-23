'use client';

import { useState } from 'react';
import { X, Search, FileText, Camera, Mic, Globe, Loader2, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react';
import axios from 'axios';

interface FactCheckerProps {
  onClose: () => void;
}

interface Claim {
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

interface Evidence {
  id: string;
  title: string;
  content: string;
  url: string;
  source_type: string;
  relevance_score: number;
  stance: string;
  confidence: number;
}

interface FactCheckResult {
  claims: Claim[];
  evidence: Evidence[];
  manipulation_detected: boolean;
  manipulation_types: string[];
  overall_verdict: string;
  confidence: number;
}

export default function FactChecker({ onClose }: FactCheckerProps) {
  const [inputType, setInputType] = useState<'text' | 'url' | 'file'>('text');
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<FactCheckResult | null>(null);
  const [error, setError] = useState('');

  const inputTypes = [
    { id: 'text', icon: <FileText className="w-5 h-5" />, label: 'Text Input' },
    { id: 'url', icon: <Globe className="w-5 h-5" />, label: 'URL Analysis' },
    { id: 'file', icon: <Camera className="w-5 h-5" />, label: 'File Upload' }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      // This would be replaced with actual API call to TruthLens backend
      const response = await axios.post('/api/fact-check', {
        input_type: inputType,
        content: input
      });

      setResult(response.data);
    } catch (err) {
      setError('Failed to process the request. Please try again.');
      console.error('Fact-check error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'true':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'false':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'misleading':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      default:
        return <Info className="w-5 h-5 text-blue-500" />;
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'true':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'false':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'misleading':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default:
        return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">TruthLens Fact Checker</h2>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <X className="w-6 h-6 text-gray-500" />
        </button>
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="mb-8">
        {/* Input Type Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Input Type
          </label>
          <div className="grid grid-cols-3 gap-4">
            {inputTypes.map((type) => (
              <button
                key={type.id}
                type="button"
                onClick={() => setInputType(type.id as any)}
                className={`p-4 border-2 rounded-lg flex flex-col items-center space-y-2 transition-all ${
                  inputType === type.id
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                {type.icon}
                <span className="text-sm font-medium">{type.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Input Field */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {inputType === 'text' && 'Enter text to fact-check'}
            {inputType === 'url' && 'Enter URL to analyze'}
            {inputType === 'file' && 'Upload file to analyze'}
          </label>
          {inputType === 'text' || inputType === 'url' ? (
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                inputType === 'text'
                  ? 'Paste the text you want to fact-check here...'
                  : 'Enter the URL you want to analyze...'
              }
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={6}
              required
            />
          ) : (
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <Camera className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">Click to upload or drag and drop</p>
              <p className="text-sm text-gray-500">Supports images, videos, and audio files</p>
              <input
                type="file"
                className="hidden"
                accept="image/*,video/*,audio/*"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setInput(file.name);
                  }
                }}
              />
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Search className="w-5 h-5" />
              <span>Fact Check</span>
            </>
          )}
        </button>
      </form>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Overall Verdict */}
          <div className="bg-white border rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Overall Verdict</h3>
            <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full border ${getVerdictColor(result.overall_verdict)}`}>
              {getVerdictIcon(result.overall_verdict)}
              <span className="font-medium">{result.overall_verdict}</span>
            </div>
            <p className="text-sm text-gray-600 mt-2">
              Confidence: {Math.round(result.confidence * 100)}%
            </p>
          </div>

          {/* Claims */}
          {result.claims.length > 0 && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Detected Claims</h3>
              <div className="space-y-4">
                {result.claims.map((claim) => (
                  <div key={claim.id} className="border-l-4 border-blue-500 pl-4">
                    <p className="font-medium text-gray-800 mb-2">{claim.text}</p>
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><span className="font-medium">Subject:</span> {claim.subject}</p>
                      <p><span className="font-medium">Predicate:</span> {claim.predicate}</p>
                      <p><span className="font-medium">Object:</span> {claim.object}</p>
                      <p><span className="font-medium">Checkworthiness:</span> {Math.round(claim.checkworthiness * 100)}%</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Evidence */}
          {result.evidence.length > 0 && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Supporting Evidence</h3>
              <div className="space-y-4">
                {result.evidence.map((evidence) => (
                  <div key={evidence.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-medium text-gray-800">{evidence.title}</h4>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        evidence.stance === 'supports' ? 'bg-green-100 text-green-800' :
                        evidence.stance === 'contradicts' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {evidence.stance}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{evidence.content}</p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <a href={evidence.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                        {evidence.url}
                      </a>
                      <span>Relevance: {Math.round(evidence.relevance_score * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Manipulation Detection */}
          {result.manipulation_detected && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4 text-red-600">Manipulation Detected</h3>
              <div className="space-y-2">
                {result.manipulation_types.map((type, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                    <span className="text-gray-700">{type}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
