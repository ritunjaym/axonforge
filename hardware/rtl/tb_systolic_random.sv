// Randomized RTL Testbench for the pipelined systolic array.
//
// Constrained-random test parameters (matching the Python driver):
//   - Activation values: random INT16 (−32767 to +32767)
//   - Activation sparsity: 0–30% zeros
//   - Weight values: random INT16
//   - Array size: parameterised (default 8×8)
//
// Usage (standalone — requires $urandom_range support):
//   iverilog -g2012 -o sim.vvp tb_systolic_random.sv systolic_array.sv
//   vvp sim.vvp
//
// The testbench prints outputs as:
//   OUT <i> <j> <value>
// where (i,j) indexes the output row/column and value is the RTL result.
// Compare these against the Python golden (hardware/systolic_array.py).
//
// Note: for the 500-case test suite, the Python driver (validate_rtl.py)
// generates deterministic testbenches with embedded random values rather than
// running this standalone SV testbench. This file serves as documentation of
// the constrained-random approach and can be run independently.
//
// Pass criterion: 500/500 cases, 0 integer-mode output deltas vs Python golden.

`timescale 1ns/1ps

module tb_systolic_random;
    // Array parameters
    localparam int ROWS        = 8;
    localparam int COLS        = 8;
    localparam int DATA_WIDTH  = 16;
    localparam int PIPE_STAGES = 0;
    localparam int ACC_WIDTH   = 40;
    localparam int N_CASES     = 10;   // number of random cases to run
    localparam int SEED        = 42;

    // Derived timing parameters
    localparam int HOP      = 1 + PIPE_STAGES;
    localparam int FILL     = (ROWS + COLS - 2) * HOP;
    localparam int STEADY   = ROWS * HOP;
    localparam int TOTAL    = FILL + STEADY + FILL + HOP;

    logic clk = 0;
    logic rst_n;

    logic signed [DATA_WIDTH-1:0] weights [ROWS][COLS];
    logic signed [DATA_WIDTH-1:0] act_in  [ROWS];
    logic signed [ACC_WIDTH-1:0]  sum_out [COLS];

    // Injection schedule and expected outputs (computed per case in initial)
    logic signed [DATA_WIDTH-1:0] inj [TOTAL][ROWS];

    systolic_array #(
        .ROWS(ROWS), .COLS(COLS),
        .DATA_WIDTH(DATA_WIDTH), .PIPE_STAGES(PIPE_STAGES), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weights(weights), .act_in(act_in), .sum_out(sum_out)
    );

    always #5 clk = ~clk;   // 100 MHz

    // VCD for waveform viewing
    initial begin
        $dumpfile("tb_systolic_random.vcd");
        $dumpvars(0, tb_systolic_random);
    end

    integer t, k, j;
    logic signed [DATA_WIDTH-1:0] A [ROWS][ROWS];
    integer case_idx;

    initial begin
        for (case_idx = 0; case_idx < N_CASES; case_idx++) begin
            // Generate random weights and activations
            for (k = 0; k < ROWS; k++)
                for (j = 0; j < COLS; j++)
                    weights[k][j] = $signed($urandom_range(0, 65535)) - 32768;

            for (int i = 0; i < ROWS; i++)
                for (k = 0; k < ROWS; k++) begin
                    // 0–30% sparsity on activations
                    if ($urandom_range(0, 99) < 30)
                        A[i][k] = 0;
                    else
                        A[i][k] = $signed($urandom_range(0, 65535)) - 32768;
                end

            // Build injection schedule (skewed wavefront)
            for (t = 0; t < TOTAL; t++)
                for (k = 0; k < ROWS; k++)
                    inj[t][k] = 0;

            for (int i = 0; i < ROWS; i++)
                for (k = 0; k < ROWS; k++) begin
                    automatic int cyc = (i + k) * HOP;
                    if (cyc < TOTAL)
                        inj[cyc][k] = A[i][k];
                end

            // Reset DUT
            rst_n = 0;
            @(posedge clk); #1;
            rst_n = 1;

            // Run simulation and capture outputs
            for (t = 0; t < TOTAL; t++) begin
                for (k = 0; k < ROWS; k++)
                    act_in[k] = inj[t][k];
                @(posedge clk); #1;

                // Output capture at cycle (i+j+ROWS-1)*HOP
                for (int i = 0; i < ROWS; i++)
                    for (j = 0; j < COLS; j++)
                        if (t == (i + j + ROWS - 1) * HOP)
                            $display("CASE %0d OUT %0d %0d %0d",
                                     case_idx, i, j, $signed(sum_out[j]));
            end
        end
        $display("DONE");
        $finish;
    end
endmodule
