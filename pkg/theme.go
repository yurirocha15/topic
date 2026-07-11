package main

import "github.com/gdamore/tcell/v2"

const (
	themeResetTag        = "[-]"
	themeStyleResetTag   = "[-:-:-]"
	themeLabelTag        = "[white]"
	themeStrongTag       = "[white::b]"
	themeMutedTag        = "[gray]"
	themeAccentTag       = "[aqua]"
	themeAttentionTag    = "[gold]"
	themeTrackTag        = "[gray]"
	usageWarningPercent  = 70.0
	usageCriticalPercent = 90.0
	usageDisplayMaximum  = 999.9
)

func usageColorName(percent float64) string {
	switch {
	case percent >= usageCriticalPercent:
		return "red"
	case percent >= usageWarningPercent:
		return "gold"
	default:
		return "green"
	}
}

func usageCellColor(percent float64) tcell.Color {
	switch {
	case percent >= usageCriticalPercent:
		return tcell.ColorRed
	case percent >= usageWarningPercent:
		return tcell.ColorGold
	default:
		return tcell.ColorGreen
	}
}
