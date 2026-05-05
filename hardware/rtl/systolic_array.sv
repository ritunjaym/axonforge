// Pipelined Systolic Array — weight-stationary dataflow
//
// Parameters:
//   ROWS        — number of PE rows    (= K, inner dimension)
//   COLS        — number of PE columns (= N, output dimension)
//   DATA_WIDTH  — bits per activation/weight element
//   PIPE_STAGES — additional pipeline register stages between PE hops
//                 (0 = 1 cycle/hop, 1 = 2 cycles/hop, etc.)
//
// Interface:
//   weights[k][j]  — static weight matrix, written before computation begins
//   act_in[k]      — activation left edge (row k), driven each cycle by controller
//   sum_out[j]     — partial sum bottom edge (column j), read at output cycles
//
// Dataflow:
//   Activations flow EAST through pipeline registers.
//   Partial sums flow SOUTH through pipeline registers.
//   PE[k][j]: sum_out = sum_in + weight[k][j] * act_in
//
// Co-simulation contract:
//   Integer-mode output must exactly match hardware/systolic_array.py golden.
//   The Python driver (validate_rtl.py) injects activations using the same
//   skewed schedule: act_in[k] = A[i][k] at cycle (i+k)*(1+PIPE_STAGES).
//
// Pitfall: compile with -g2012 for always_ff / always_comb support.

`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// Single PE with pipeline registers on east and south outputs
// ---------------------------------------------------------------------------

// ACC_WIDTH must be wide enough to hold ROWS × INT16_MAX²:
//   8 × (2^15-1)² ≈ 8.6×10⁹ requires 34-bit signed.
//   Default 40 bits handles up to ROWS=256 without overflow.
module pe #(
    parameter int DATA_WIDTH  = 16,
    parameter int PIPE_STAGES = 0,
    parameter int ACC_WIDTH   = 40
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic signed [DATA_WIDTH-1:0] weight,
    input  logic signed [DATA_WIDTH-1:0] act_in,
    input  logic signed [ACC_WIDTH-1:0]  sum_in,
    output logic signed [DATA_WIDTH-1:0] act_out,   // east (to PE[k][j+1])
    output logic signed [ACC_WIDTH-1:0]  sum_out    // south (to PE[k+1][j])
);
    // Compute MAC combinationally
    logic signed [ACC_WIDTH-1:0] mac_result;
    always_comb mac_result = sum_in + ACC_WIDTH'(DATA_WIDTH'(weight) * DATA_WIDTH'(act_in));

    // Pipeline register chain for east (activation pass-through)
    // Depth = 1 + PIPE_STAGES to give hop_latency = 1 + PIPE_STAGES cycles
    logic signed [DATA_WIDTH-1:0] act_pipe [0:PIPE_STAGES];
    logic signed [ACC_WIDTH-1:0]  sum_pipe [0:PIPE_STAGES];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s <= PIPE_STAGES; s++) begin
                act_pipe[s] <= '0;
                sum_pipe[s] <= '0;
            end
        end else begin
            act_pipe[0] <= act_in;
            sum_pipe[0] <= mac_result;
            for (int s = 1; s <= PIPE_STAGES; s++) begin
                act_pipe[s] <= act_pipe[s-1];
                sum_pipe[s] <= sum_pipe[s-1];
            end
        end
    end

    assign act_out = act_pipe[PIPE_STAGES];
    assign sum_out = sum_pipe[PIPE_STAGES];
endmodule


// ---------------------------------------------------------------------------
// Systolic array top: ROWS × COLS grid of PEs
// ---------------------------------------------------------------------------

module systolic_array #(
    parameter int ROWS        = 8,
    parameter int COLS        = 8,
    parameter int DATA_WIDTH  = 16,
    parameter int PIPE_STAGES = 0,
    parameter int ACC_WIDTH   = 40   // wide enough for ROWS × INT16_MAX²
) (
    input  logic                        clk,
    input  logic                        rst_n,
    // Weight matrix (written once before computation)
    input  logic signed [DATA_WIDTH-1:0] weights [ROWS][COLS],
    // Activation left edge: one value per row per cycle
    input  logic signed [DATA_WIDTH-1:0] act_in  [ROWS],
    // Partial sum bottom edge: output of the last row
    output logic signed [ACC_WIDTH-1:0]  sum_out [COLS]
);

    // Horizontal wires: act_wire[k][j] drives PE[k][j]'s act_in
    // act_wire[k][0] = act_in[k] (external left edge)
    // act_wire[k][j+1] = PE[k][j].act_out
    logic signed [DATA_WIDTH-1:0] act_wire [ROWS][COLS+1];

    // Vertical wires: sum_wire[k][j] drives PE[k][j]'s sum_in
    // sum_wire[0][j] = 0 (top edge)
    // sum_wire[k+1][j] = PE[k][j].sum_out
    logic signed [ACC_WIDTH-1:0] sum_wire [ROWS+1][COLS];

    // Connect left edge to external input
    genvar gk;
    generate
        for (gk = 0; gk < ROWS; gk++) begin : g_left_edge
            assign act_wire[gk][0] = act_in[gk];
        end
    endgenerate

    // Top edge: zero initial partial sums
    genvar gj;
    generate
        for (gj = 0; gj < COLS; gj++) begin : g_top_edge
            assign sum_wire[0][gj] = '0;
        end
    endgenerate

    // Instantiate PE grid
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : g_row
            for (c = 0; c < COLS; c++) begin : g_col
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .PIPE_STAGES(PIPE_STAGES),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .weight (weights[r][c]),
                    .act_in (act_wire[r][c]),
                    .sum_in (sum_wire[r][c]),
                    .act_out(act_wire[r][c+1]),
                    .sum_out(sum_wire[r+1][c])
                );
            end
        end
    endgenerate

    // Bottom edge: array output
    generate
        for (gj = 0; gj < COLS; gj++) begin : g_bottom_edge
            assign sum_out[gj] = sum_wire[ROWS][gj];
        end
    endgenerate

endmodule
