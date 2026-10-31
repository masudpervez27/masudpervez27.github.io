document.addEventListener("DOMContentLoaded", function () {
  var codeBlocks = document.querySelectorAll("pre > code.language-mermaid");
  if (!codeBlocks.length) return;

  if (!window.mermaid) {
    console.error("Mermaid could not be loaded; diagrams remain visible as code.");
    return;
  }

  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
  });

  var diagrams = [];
  codeBlocks.forEach(function (codeBlock) {
    var container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = codeBlock.textContent;
    codeBlock.parentElement.parentElement.replaceWith(container);
    diagrams.push(container);
  });

  window.mermaid.run({ nodes: diagrams }).catch(function (error) {
    console.error("Mermaid diagram rendering failed.", error);
  });
});
