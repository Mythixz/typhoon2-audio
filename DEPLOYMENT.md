# 🚀 Deployment Guide - Vercel

## 📋 **การ Deploy ขึ้น Vercel**

### **วิธีที่ 1: Deploy ผ่าน Vercel Dashboard (แนะนำ)**

1. **เข้าไปที่ [vercel.com](https://vercel.com)**
2. **Sign in/Sign up** ด้วย GitHub, GitLab, หรือ Bitbucket
3. **คลิก "New Project"**
4. **Import Git Repository** เลือก repository ของคุณ
5. **ตั้งค่า Project:**
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

### **วิธีที่ 2: Deploy ผ่าน Vercel CLI**

```bash
# ติดตั้ง Vercel CLI
npm i -g vercel

# เข้าไปใน frontend folder
cd frontend

# Login Vercel
vercel login

# Deploy
vercel

# หรือ deploy ไป production
vercel --prod
```

### **วิธีที่ 3: Deploy ผ่าน GitHub Actions**

สร้างไฟล์ `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Vercel
on:
  push:
    branches: [main]
    paths: ['frontend/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: cd frontend && npm ci
      
      - name: Build
        run: cd frontend && npm run build
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          working-directory: ./frontend
```

## 🔧 **Environment Variables ที่ต้องตั้งค่า**

### **ใน Vercel Dashboard:**
1. ไปที่ Project Settings
2. เลือก Environment Variables
3. เพิ่มตัวแปรต่อไปนี้:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.com
NEXT_PUBLIC_APP_NAME=Accessibility-First Call Center System
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_DEMO_MODE=true
```

### **Production Environment Variables:**
```env
NEXT_PUBLIC_API_BASE_URL=https://your-production-backend.com
NEXT_PUBLIC_APP_NAME=Accessibility-First Call Center System
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_DEMO_MODE=false
```

## 🚨 **สิ่งที่ต้องระวัง**

### **1. API Base URL**
- ต้องเปลี่ยนจาก `localhost:8000` เป็น URL จริง
- ใช้ HTTPS ใน production
- ตั้งค่า CORS ใน backend

### **2. Build Errors**
- ตรวจสอบ dependencies ใน package.json
- ใช้ Node.js version ที่เหมาะสม (18+)
- ตรวจสอบ TypeScript errors

### **3. Performance**
- ใช้ `next build` ก่อน deploy
- ตรวจสอบ bundle size
- ใช้ Vercel Analytics

## 📊 **การ Monitor และ Debug**

### **Vercel Dashboard:**
- **Functions**: ดู serverless functions
- **Analytics**: ดู performance metrics
- **Logs**: ดู error logs

### **Local Testing:**
```bash
# Build production
npm run build

# Start production server
npm start

# Test production build
npm run lint
```

## 🔄 **การ Update และ Redeploy**

### **Automatic Deploy:**
- Push ไป main branch = auto deploy
- Vercel จะ build และ deploy อัตโนมัติ

### **Manual Deploy:**
```bash
# ใน Vercel Dashboard
# หรือใช้ CLI
vercel --prod
```

## 📱 **Custom Domain**

1. ไปที่ Project Settings > Domains
2. เพิ่ม custom domain
3. ตั้งค่า DNS records
4. ใช้ Vercel DNS หรือ external DNS

---

## 🎯 **สรุป**

**Frontend พร้อม deploy แล้ว!** 

### **สิ่งที่ต้องทำ:**
1. ✅ สร้างไฟล์ config ครบแล้ว
2. 🔧 ตั้งค่า Environment Variables
3. 🚀 Deploy ผ่าน Vercel Dashboard
4. 🔗 เชื่อมต่อกับ Backend

### **ผลลัพธ์:**
- **URL**: `https://your-project.vercel.app`
- **Performance**: Fast, Optimized
- **Scalability**: Auto-scaling
- **Monitoring**: Built-in analytics

**🎉 ตอนนี้คุณมี Frontend ที่ deploy บน Vercel แล้ว!** 