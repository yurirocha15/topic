package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// TestExportedDeclarationsHaveDocComments keeps every exported type, function, variable, and
// constant documented, along with exported methods of exported types. revive skips this check in
// package main, so the test enforces it. Files for every build tag are parsed.
func TestExportedDeclarationsHaveDocComments(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var missing []string
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, parseErr := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		for _, decl := range file.Decls {
			missing = append(missing, undocumented(fset, decl)...)
		}
	}
	if len(missing) > 0 {
		t.Errorf("exported declarations without a doc comment:\n%s", strings.Join(missing, "\n"))
	}
}

func undocumented(fset *token.FileSet, decl ast.Decl) []string {
	switch decl := decl.(type) {
	case *ast.FuncDecl:
		if undocumentedFunc(decl) {
			return []string{fset.Position(decl.Pos()).String() + " " + decl.Name.Name}
		}
	case *ast.GenDecl:
		if decl.Doc != nil {
			return nil
		}
		var missing []string
		for _, spec := range decl.Specs {
			for _, ident := range undocumentedSpec(spec) {
				missing = append(missing, fset.Position(ident.Pos()).String()+" "+ident.Name)
			}
		}
		return missing
	}
	return nil
}

// undocumentedFunc reports an exported function, or an exported method of an exported type,
// without a doc comment.
func undocumentedFunc(decl *ast.FuncDecl) bool {
	if decl.Doc != nil || !decl.Name.IsExported() {
		return false
	}
	return decl.Recv == nil || ast.IsExported(receiverTypeName(decl.Recv))
}

// undocumentedSpec returns the exported names a spec declares without a doc comment. A grouped
// block can be documented once for all of its values.
func undocumentedSpec(spec ast.Spec) []*ast.Ident {
	switch spec := spec.(type) {
	case *ast.TypeSpec:
		if spec.Name.IsExported() && spec.Doc == nil {
			return []*ast.Ident{spec.Name}
		}
	case *ast.ValueSpec:
		if spec.Doc != nil || spec.Comment != nil {
			return nil
		}
		var missing []*ast.Ident
		for _, name := range spec.Names {
			if name.IsExported() {
				missing = append(missing, name)
			}
		}
		return missing
	}
	return nil
}

func receiverTypeName(receiver *ast.FieldList) string {
	if receiver == nil || len(receiver.List) == 0 {
		return ""
	}
	expr := receiver.List[0].Type
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	if generic, ok := expr.(*ast.IndexExpr); ok {
		expr = generic.X
	}
	if ident, ok := expr.(*ast.Ident); ok {
		return ident.Name
	}
	return ""
}
