'use client';

import { useState } from 'react';
import { Search, Shield, AlertTriangle, CheckCircle, Globe, FileText, Mic, Camera } from 'lucide-react';
import FactChecker from '@/components/FactChecker';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function Home() {
  const [showFactChecker, setShowFactChecker] = useState(false);

  const features = [
    {
      icon: <Search className="w-6 h-6" />,
      title: "Evidence-Based Verification",
      description: "Cross-references claims with trusted sources including government websites, news organizations, and academic databases."
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Manipulation Detection",
      description: "Identifies common misinformation tactics and manipulation techniques used in content."
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: "Multilingual Support",
      description: "Processes content in multiple languages including English, Hindi, Tamil, Telugu, and more."
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: "Multi-Modal Input",
      description: "Accepts text, images, videos, and audio files for comprehensive fact-checking."
    }
  ];

  const inputTypes = [
    { icon: <FileText className="w-5 h-5" />, label: "Text Input" },
    { icon: <Camera className="w-5 h-5" />, label: "Image Upload" },
    { icon: <Mic className="w-5 h-5" />, label: "Audio/Video" },
    { icon: <Globe className="w-5 h-5" />, label: "URL Analysis" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center py-16">
          <div className="flex items-center justify-center mb-6">
            <Shield className="w-12 h-12 text-blue-600 mr-3" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              TruthLens
            </h1>
          </div>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            A comprehensive fact-checking and misinformation detection system that processes multilingual and multimedia content with AI-powered verification.
          </p>
          
          <button
            onClick={() => setShowFactChecker(true)}
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            Start Fact-Checking
          </button>
        </div>

        {/* Features Section */}
        <div className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-800">
            Advanced Features
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-200">
                <div className="text-blue-600 mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-3 text-gray-800">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Input Types Section */}
        <div className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-800">
            Multiple Input Types Supported
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {inputTypes.map((type, index) => (
              <div key={index} className="bg-white p-6 rounded-xl shadow-md text-center hover:shadow-lg transition-shadow duration-200">
                <div className="text-blue-600 mb-3 flex justify-center">{type.icon}</div>
                <p className="font-semibold text-gray-800">{type.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Trusted Sources Section */}
        <div className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-800">
            Trusted Sources
          </h2>
          <div className="bg-white p-8 rounded-xl shadow-md">
            <div className="grid md:grid-cols-3 gap-8">
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800">Government Sources</h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• World Health Organization (WHO)</li>
                  <li>• Centers for Disease Control (CDC)</li>
                  <li>• Food and Drug Administration (FDA)</li>
                  <li>• National Institutes of Health (NIH)</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800">News Organizations</h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• Reuters</li>
                  <li>• Associated Press (AP)</li>
                  <li>• BBC News</li>
                  <li>• Press Information Bureau (PIB)</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800">Reference Sources</h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• Wikipedia</li>
                  <li>• Academic Databases</li>
                  <li>• Peer-reviewed Journals</li>
                  <li>• Fact-checking Organizations</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Fact Checker Modal */}
      {showFactChecker && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <FactChecker onClose={() => setShowFactChecker(false)} />
          </div>
        </div>
      )}

      <Footer />
    </div>
  );
}
