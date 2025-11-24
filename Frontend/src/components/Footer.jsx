import React from 'react';
import { Sprout } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-8 mt-16">
      <div className="max-w-7xl mx-auto px-4 text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Sprout className="w-6 h-6 text-green-400" />
          <span className="text-xl font-bold">CropPred</span>
        </div>
        <p className="text-gray-400">Empowering farmers with AI-driven yield predictions</p>
        <p className="text-gray-500 text-sm mt-4">© {new Date().getFullYear()} CropPred. All rights reserved.</p>
      </div>
    </footer>
  );
}
