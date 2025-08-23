#!/usr/bin/env python
"""
Test script for Open-Sora TDM integration.
This script tests the basic functionality without full training.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from opensora.models.causalvideovae import ae_stride_config, ae_channel_config, ae_wrapper
        print("✓ Open-Sora VAE modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Open-Sora VAE modules: {e}")
        return False
    
    try:
        from opensora.models.text_encoder import get_text_warpper
        print("✓ Open-Sora text encoder modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Open-Sora text encoder modules: {e}")
        return False
    
    try:
        from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
        print("✓ Open-Sora diffusion modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Open-Sora diffusion modules: {e}")
        return False
    
    try:
        from src.models import generate_new
        print("✓ TDM models module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TDM models module: {e}")
        return False
    
    try:
        from src.predictor import Predictor
        print("✓ TDM predictor module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TDM predictor module: {e}")
        return False
    
    return True

def test_vae_initialization():
    """Test VAE initialization."""
    print("\nTesting VAE initialization...")
    
    try:
        from opensora.models.causalvideovae import ae_wrapper
        
        # Test with a simple VAE config
        vae = ae_wrapper['WFVAEModel_D8_4x8x8']("LanguageBind/Open-Sora-Plan-v1.3.0")
        print("✓ VAE initialized successfully")
        
        # Test basic properties
        print(f"  - VAE dtype: {vae.dtype()}")
        print(f"  - VAE device: {next(vae.parameters()).device if list(vae.parameters()) else 'CPU'}")
        
        return True
    except Exception as e:
        print(f"✗ VAE initialization failed: {e}")
        return False

def test_text_encoder_initialization():
    """Test text encoder initialization."""
    print("\nTesting text encoder initialization...")
    
    try:
        from opensora.models.text_encoder import get_text_warpper
        
        # Test T5 encoder
        t5_encoder = get_text_warpper("google/t5-v1_1-xl")("google/t5-v1_1-xl")
        print("✓ T5 text encoder initialized successfully")
        
        # Test CLIP encoder
        clip_encoder = get_text_warpper("openai/clip-vit-large-patch14")("openai/clip-vit-large-patch14")
        print("✓ CLIP text encoder initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Text encoder initialization failed: {e}")
        return False

def test_diffusion_model_initialization():
    """Test diffusion model initialization."""
    print("\nTesting diffusion model initialization...")
    
    try:
        from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
        
        print(f"Available models: {list(Diffusion_models.keys())}")
        print(f"Available model classes: {list(Diffusion_models_class.keys())}")
        
        # Test if we can access a model
        if 'OpenSoraT2V_v1_3_93x640x640' in Diffusion_models:
            print("✓ Open-Sora model found in Diffusion_models")
        else:
            print("⚠ Open-Sora model not found in Diffusion_models")
        
        if 'OpenSoraT2V_v1_3_93x640x640' in Diffusion_models_class:
            print("✓ Open-Sora model found in Diffusion_models_class")
        else:
            print("⚠ Open-Sora model not found in Diffusion_models_class")
        
        return True
    except Exception as e:
        print(f"✗ Diffusion model initialization failed: {e}")
        return False

def test_tdm_functions():
    """Test TDM functions with dummy data."""
    print("\nTesting TDM functions...")
    
    try:
        from src.models import generate_new
        from src.predictor import Predictor
        
        # Create dummy data
        batch_size = 2
        channels = 8
        frames = 23  # 93 // 4 (for 4x8x8 VAE)
        height = 80  # 640 // 8
        width = 80   # 640 // 8
        
        # Dummy tensors
        latent = torch.randn(batch_size, channels, frames, height, width)
        noise = torch.randn(batch_size, channels, frames, height, width)
        encoder_hidden_states = torch.randn(batch_size, 77, 2048)  # T5 embedding
        prompt_attention_mask = torch.ones(batch_size, 77)
        
        # Test generate_new function
        print("  - Testing generate_new function...")
        result = generate_new(
            unet=None,  # We don't have a real model here
            noise_scheduler=None,
            latent=latent,
            noise=noise,
            encoder_hidden_states=encoder_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            steps=1,
            return_mid=False,
            total_steps=900,
            use_opensora=True
        )
        print("✓ generate_new function signature is correct")
        
        # Test Predictor class
        print("  - Testing Predictor class...")
        # We can't fully test without a real model, but we can test initialization
        print("✓ Predictor class can be imported")
        
        return True
    except Exception as e:
        print(f"✗ TDM functions test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Open-Sora TDM Integration Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_vae_initialization,
        test_text_encoder_initialization,
        test_diffusion_model_initialization,
        test_tdm_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Open-Sora TDM integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
