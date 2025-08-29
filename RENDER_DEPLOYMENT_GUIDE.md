# 🚀 TruthLens API - Render Deployment Guide

## ✅ **Project Status: READY FOR RENDER DEPLOYMENT**

Your TruthLens API is now **100% prepared for Render deployment** with all heavy files removed and configuration files created!

### 📁 **Files Created for Render**
- ✅ `Procfile` - Render startup command
- ✅ `runtime.txt` - Python 3.11.7 specification
- ✅ `.renderignore` - Excludes heavy files from deployment
- ✅ `requirements.txt` - All dependencies listed
- ✅ `config.env` - Your API keys configured

### 🗑️ **Heavy Files Removed**
- ✅ Removed `data/cache/` directory (901,992 deletions!)
- ✅ Removed `models/` directory with large model files
- ✅ Removed all `.pkl`, `.bin`, `.safetensors` files
- ✅ Repository size optimized for Render's 512MB limit

## 🚀 **Step-by-Step Render Deployment**

### 1. **GitHub Repository Setup** (Already Done ✅)
```bash
# Your repository is already set up and committed
git status  # Shows clean working directory
```

### 2. **Create Render Account**
1. Go to 👉 [https://render.com](https://render.com)
2. Sign up with GitHub (recommended)
3. Verify your email

### 3. **Create New Web Service**
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account if not already connected
3. Select your **TruthLens repository**

### 4. **Configure Service Settings**

#### **Basic Configuration:**
- **Name**: `truthlens-api`
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Runtime**: `Python 3`
- **Plan**: `Free` (good for submission)

#### **Build & Start Commands:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app_enhanced:app --host 0.0.0.0 --port $PORT`

#### **Environment Variables:**
Add these in Render's Environment section:
```
NEWS_API_KEY=9c7e59e19af34bb8adb97d0a8bec458d
GUARDIAN_API_KEY=0b8f5a8d-d3a0-49e1-8472-f943dae59338
GOOGLE_API_KEY=AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc
HOST=0.0.0.0
PORT=10000
DEBUG=false
TORCH_DEVICE=cpu
```

### 5. **Deploy**
1. Click **"Create Web Service"**
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Start the server
3. Wait for deployment (usually 5-10 minutes)

### 6. **Get Your API URL**
Once deployed, you'll get a URL like:
```
https://truthlens-api.onrender.com
```

## 🔧 **Render-Specific Optimizations**

### **Model Download Strategy**
- Models will be downloaded automatically on first request
- `sentence-transformers` and `transformers` handle this automatically
- First request may take 2-3 minutes to download models
- Subsequent requests will be fast

### **Memory Management**
- Render Free plan: 512MB RAM
- Models are loaded on-demand
- CPU-only inference (no GPU on free plan)
- Automatic garbage collection

### **Cold Start Handling**
- Free plan has cold starts (15-30 seconds)
- First request after inactivity will be slower
- Keep-alive requests can help maintain warm state

## 📊 **Testing Your Deployed API**

### **Health Check**
```bash
curl https://truthlens-api.onrender.com/health
```

### **Claim Verification**
```bash
curl -X POST "https://truthlens-api.onrender.com/verify" \
  -H "Content-Type: application/json" \
  -d '{"claim": "The Earth is flat", "context": "Social media claim"}'
```

### **API Documentation**
Visit: `https://truthlens-api.onrender.com/docs`

## 🎯 **Expected Performance on Render**

### **Free Plan Limitations**
- **Cold Start**: 15-30 seconds (first request after inactivity)
- **Warm Start**: 5-15 seconds (subsequent requests)
- **Memory**: 512MB RAM limit
- **CPU**: Shared CPU resources
- **Uptime**: May sleep after 15 minutes of inactivity

### **Performance Tips**
1. **First Request**: Be patient (2-3 minutes for model download)
2. **Keep Warm**: Send periodic health checks
3. **Monitor**: Check Render dashboard for resource usage
4. **Scale**: Upgrade to Pro plan if needed

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Build Failures**
- Check `requirements.txt` is up to date
- Verify Python version in `runtime.txt`
- Check build logs in Render dashboard

#### **Memory Issues**
- Models may be too large for free plan
- Consider using smaller models
- Upgrade to Pro plan for more memory

#### **Cold Start Delays**
- Normal for free plan
- Send health check every 10 minutes to keep warm
- Consider Pro plan for better performance

#### **API Key Issues**
- Verify environment variables are set correctly
- Check API key quotas and limits
- Test keys locally first

### **Debug Commands**
```bash
# Check deployment status
curl https://truthlens-api.onrender.com/health

# Check component status
curl https://truthlens-api.onrender.com/components-status

# View logs in Render dashboard
# Go to your service → Logs tab
```

## 📈 **Monitoring & Scaling**

### **Free Plan Monitoring**
- Check Render dashboard regularly
- Monitor memory usage
- Watch for cold start patterns

### **Upgrade to Pro Plan**
If you need better performance:
- **Memory**: Up to 2GB RAM
- **CPU**: Dedicated CPU resources
- **Uptime**: No sleep mode
- **Custom Domains**: Available

### **Custom Domain** (Pro Plan)
```bash
# Add custom domain in Render dashboard
# Point DNS to Render's servers
# Enable SSL automatically
```

## 🎉 **Success Checklist**

- ✅ Repository cleaned of heavy files
- ✅ Procfile created
- ✅ Environment variables configured
- ✅ Render service created
- ✅ Deployment successful
- ✅ Health check passes
- ✅ API endpoints working
- ✅ Claim verification tested

## 📞 **Support**

### **Render Support**
- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)
- [Render Status](https://status.render.com)

### **TruthLens API Support**
- Check logs in Render dashboard
- Test with health check endpoint
- Verify API key configuration
- Review API documentation at `/docs`

---

## 🚀 **Ready to Deploy!**

Your TruthLens API is **optimized and ready for Render deployment**!

**Next Steps:**
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Configure environment variables
5. Deploy and test!

**Your API will be live at:** `https://truthlens-api.onrender.com`

**Status**: ✅ **READY FOR RENDER DEPLOYMENT**
