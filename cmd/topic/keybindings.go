package main

import (
	"fmt"
	"strings"
)

// The key list below is the single source for the help screen and the side panel's key hints.
// README.md lists the same keys; TestReadmeControlsMatchKeyBindings keeps it in step.

type keyBinding struct {
	keys   string
	action string
	// hintGroup and hint put the binding in the side panel, as "hintKey hint" on the group's line.
	// hintKey defaults to keys.
	hintGroup string
	hintKey   string
	hint      string
}

type keySection struct {
	title    string
	bindings []keyBinding
}

const (
	hintGroupKeys  = "Keys"
	hintGroupModes = "Modes"
	hintGroupView  = "View"
)

func keySections() []keySection {
	return []keySection{
		{title: "General", bindings: []keyBinding{
			{keys: "q / Ctrl+C", action: "quit", hintGroup: hintGroupKeys, hintKey: "q", hint: "quit"},
			{keys: "?", action: "show or close the key help", hintGroup: hintGroupKeys, hint: "help"},
			{keys: "p", action: "pause or resume refreshing", hintGroup: hintGroupKeys, hint: "pause"},
			{keys: "a", action: "show or hide the logo", hintGroup: hintGroupView, hint: "logo"},
		}},
		{title: "Process table", bindings: []keyBinding{
			{keys: "↑↓ PgUp PgDn Home End", action: "move the highlight (or click a row)"},
			{keys: "← →", action: "scroll the table sideways"},
			{keys: "Enter", action: "details of the highlighted process", hintGroup: hintGroupView, hint: "details"},
			{keys: "k", action: "signal the highlighted process", hintGroup: hintGroupView, hint: "signal"},
			{keys: "/", action: "filter by command, user, or PID", hintGroup: hintGroupModes, hint: "filter"},
			{keys: "s", action: "choose the sort column and direction", hintGroup: hintGroupModes, hint: "sort"},
			{keys: "r", action: "reverse the sort", hintGroup: hintGroupModes, hint: "reverse"},
			{keys: "t", action: "show or hide the process tree", hintGroup: hintGroupModes, hint: "tree"},
		}},
		{title: "Filter mode", bindings: []keyBinding{
			{keys: "Backspace", action: "delete the last character"},
			{keys: "Ctrl+U", action: "clear the filter"},
			{keys: "Enter / Esc", action: "stop typing and keep the filter"},
		}},
		{title: "Sort mode", bindings: []keyBinding{
			{keys: "← →", action: "choose the sort column"},
			{keys: "↓ / ↑", action: "sort high to low / low to high"},
			{keys: "Enter / Esc", action: "leave sort mode"},
		}},
		{title: "Dialogs", bindings: []keyBinding{
			{keys: "← → / Tab", action: "choose a button (starts on Cancel)"},
			{keys: "Enter", action: "press the chosen button"},
			{keys: "Esc / q", action: "close the dialog"},
		}},
	}
}

// helpText lists every key, one per line, grouped by where it works.
func helpText() string {
	sections := keySections()
	keyWidth := 0
	for _, section := range sections {
		for _, binding := range section.bindings {
			keyWidth = max(keyWidth, plainCellWidth(binding.keys))
		}
	}
	var builder strings.Builder
	for index, section := range sections {
		if index > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(section.title)
		builder.WriteByte('\n')
		for _, binding := range section.bindings {
			fmt.Fprintf(&builder, "  %s  %s\n", padPlainText(binding.keys, keyWidth), binding.action)
		}
	}
	return strings.TrimSuffix(builder.String(), "\n")
}

// keyHintLines are the side panel's short key hints, one line per group.
func keyHintLines() []string {
	groups := []string{hintGroupKeys, hintGroupModes, hintGroupView}
	hints := make(map[string][]string, len(groups))
	for _, section := range keySections() {
		for _, binding := range section.bindings {
			if binding.hintGroup == "" {
				continue
			}
			key := binding.hintKey
			if key == "" {
				key = binding.keys
			}
			hints[binding.hintGroup] = append(hints[binding.hintGroup], key+" "+binding.hint)
		}
	}
	lines := make([]string, 0, len(groups))
	for _, group := range groups {
		lines = append(lines, themeMutedTag+padPlainText(group, len(hintGroupModes)+1)+themeResetTag+
			strings.Join(hints[group], "  "))
	}
	return lines
}

// readmeControlsTable is the Markdown table README.md shows under Controls.
func readmeControlsTable() string {
	var builder strings.Builder
	builder.WriteString("| Where | Keys | Action |\n| --- | --- | --- |\n")
	for _, section := range keySections() {
		for _, binding := range section.bindings {
			fmt.Fprintf(&builder, "| %s | `%s` | %s |\n", section.title, binding.keys, binding.action)
		}
	}
	return builder.String()
}
