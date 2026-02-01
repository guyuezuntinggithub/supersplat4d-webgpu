git config user.name "xiefei.127"
git config user.email "xiefei.127@jd.com"

加载动态ply,导出动态ply,导出sog4d

你可以给这个网页的UI加一个导出单帧的功能吗？然后可以接受起始和终止帧两个参数，比如输入0-60，就是将0-60帧都导出单帧。

修改功能。

慢放。

这段 Markdown 的逻辑是清晰的，但存在一些排版细节问题，比如**中英文混排缺少空格**、**代码块语言标识不统一**以及**特殊字符（如全角空格）可能导致的代码运行错误**。

以下是修正后的版本，不仅格式更规范，还增加了步骤引导，使其更易读：

---

## 初始模型加载步骤

### 1. 准备模型文件

将您的新模型文件（例如 `your_model.sog4d`）复制到项目的 `public/` 文件夹中。

### 2. 修改源码配置

打开 `src/main.ts`，找到第 **296-298** 行，将文件名替换为您的实际文件名：

```typescript
await events.invoke('import', [{ 
    filename: '你的文件名.sog4d', 
    url: './你的文件名.sog4d' 
}]);

```

### 3. 修改构建配置

打开 `rollup.config.mjs`，找到第 **68** 行，添加或修改资源路径：

```javascript
{ src: 'public/你的文件名.sog4d' }

```

### 4. 部署上线

1. 在终端运行构建命令：
```bash
npm run build

```


2. 构建完成后，将生成的 `dist` 文件夹拖入 [Netlify Drop](https://app.netlify.com/drop) 进行部署。





