async function renderMermaidDiagrams() {
  var codeBlocks = document.querySelectorAll("pre > code.language-mermaid");
  if (!codeBlocks.length) return;

  try {
    var module = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
    var mermaid = module.default;

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
    });

    var diagrams = [];
    codeBlocks.forEach(function (codeBlock) {
      var container = document.createElement("div");
      var renderTarget = codeBlock.closest(".language-mermaid.highlighter-rouge") || codeBlock.closest("pre");
      container.className = "mermaid";
      container.textContent = codeBlock.textContent;
      renderTarget.replaceWith(container);
      diagrams.push(container);
    });

    await mermaid.run({ nodes: diagrams });
  } catch (error) {
    console.error("Mermaid could not be loaded or a diagram failed to render.", error);
  }
}

renderMermaidDiagrams();
