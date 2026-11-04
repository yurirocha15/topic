package main

import (
	"strings"

	"github.com/rivo/tview"
)

const (
	// narrowDashboardWidth is the terminal width below which the side panel is hidden, so resource
	// rows keep their labels and details.
	narrowDashboardWidth = 100
	// minTopPanelHeight keeps the borders and at least four resource rows on short terminals.
	minTopPanelHeight = 6
	// topPanelScreenShare is the largest share of the screen the top panel may take: 1/topPanelScreenShare.
	topPanelScreenShare = 2
	// narrowHelpHint is added to the process table title when the side panel and its key hints are hidden.
	narrowHelpHint = "? help "
)

// dashboard holds the main screen: resources and the side panel on top, processes below.
type dashboard struct {
	state        *State
	resourceView *tview.TextView
	infoView     *tview.TextView
	processTable *tview.Table
	topPanel     *tview.Flex
	mainLayout   *tview.Flex
	pages        *tview.Pages

	// renderedWidth and renderedHeight are the screen size the last layout used.
	renderedWidth  int
	renderedHeight int
}

func newDashboard(state *State) *dashboard {
	d := &dashboard{
		state:        state,
		resourceView: tview.NewTextView().SetDynamicColors(true).SetWrap(false),
		infoView:     tview.NewTextView().SetDynamicColors(true).SetTextAlign(tview.AlignLeft),
		processTable: configureProcessTable(tview.NewTable()),
	}
	trackProcessSelection(d.processTable, state)
	d.topPanel = tview.NewFlex().
		SetDirection(tview.FlexColumn).
		AddItem(d.resourceView, 0, 1, false).
		AddItem(d.infoView, compactInfoPanelWidth, 0, false)
	d.topPanel.SetBorder(true).SetTitle(" System ")
	d.mainLayout = tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(d.topPanel, placeholderHeight, 0, false).
		AddItem(d.processTable, 0, 1, true)
	d.pages = tview.NewPages().AddPage("main", d.mainLayout, true, true)
	return d
}

// render fills the dashboard and sizes its panels for the current screen size.
func (d *dashboard) render() {
	_, _, width, height := d.pages.GetRect()
	d.renderedWidth, d.renderedHeight = width, height

	infoWidth := infoPanelWidth(d.state, width)
	d.topPanel.ResizeItem(d.infoView, infoWidth, 0)
	infoHeight := 0
	if infoWidth > 0 {
		infoHeight = updateInfoView(d.infoView, d.state)
	} else {
		d.infoView.Clear()
	}

	// The resource view is the top panel's inner width minus the side panel and the view's padding.
	resourceWidth := width - borderHeight - infoWidth - borderHeight
	maxRows := maxTopPanelRows(height)
	resourceHeight := updateResourceView(d.resourceView, d.state, resourceWidth, maxRows)
	topRows := max(resourceHeight, infoHeight)
	if maxRows > 0 {
		topRows = min(topRows, maxRows)
	}
	d.mainLayout.ResizeItem(d.topPanel, topRows+borderHeight, 0)

	updateProcessTable(d.processTable, d.state)
	if infoWidth == 0 && width > 0 {
		d.processTable.SetTitle(d.processTable.GetTitle() + narrowHelpHint)
	}
}

// infoPanelWidth is the side panel's width, or 0 when the terminal is too narrow for it.
func infoPanelWidth(state *State, screenWidth int) int {
	if screenWidth > 0 && screenWidth < narrowDashboardWidth {
		return 0
	}
	return currentInfoPanelWidth(state)
}

// maxTopPanelRows limits the top panel to half the screen, so the process table keeps the other
// half. It returns 0, meaning no limit, before the screen size is known.
func maxTopPanelRows(screenHeight int) int {
	if screenHeight <= 0 {
		return 0
	}
	return max(minTopPanelHeight, screenHeight/topPanelScreenShare) - borderHeight
}

// updateResourceView fills the resource view for the given width, using at most maxRows rows when
// maxRows is positive, and returns the number of rows used.
func updateResourceView(view *tview.TextView, state *State, width int, maxRows int) int {
	state.dynamic.mu.Lock()
	text := buildResourceTextWithLimit(width, maxRows, state)
	state.dynamic.mu.Unlock()
	view.SetText(text)
	return strings.Count(text, "\n")
}
