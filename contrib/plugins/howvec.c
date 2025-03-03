/*
 * Copyright (C) 2019, Alex Bennée <alex.bennee@linaro.org>
 *
 * How vectorised is this code?
 *
 * Attempt to measure the amount of vectorisation that has been done
 * on some code by counting classes of instruction.
 *
 * License: GNU GPL, version 2 or later.
 *   See the COPYING file in the top-level directory.
 */
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <glib.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

typedef enum {
    COUNT_CLASS,
    COUNT_INDIVIDUAL,
    COUNT_NONE
} CountType;

static int limit = 50;
static bool do_inline;
static bool verbose;

static GMutex lock;
static GHashTable *insns;

typedef struct {
    const char *class;
    const char *opt;
    uint32_t mask;
    uint32_t pattern;
    CountType what;
    uint64_t count;
} InsnClassExecCount;

typedef struct {
    char *insn;
    uint32_t opcode;
    uint64_t count;
    InsnClassExecCount *class;
} InsnExecCount;

/*
 * Matchers for classes of instructions, order is important.
 *
 * Your most precise match must be before looser matches. If no match
 * is found in the table we can create an individual entry.
 *
 * 31..28 27..24 23..20 19..16 15..12 11..8 7..4 3..0
 */
static InsnClassExecCount aarch64_insn_classes[] = {
    /* "Reserved"" */
    { "  SVE",               "sve",    0xf5000000, 0x65000000, COUNT_CLASS},
    { "  SVE",               "sve",    0xf5000000, 0x25000000, COUNT_CLASS},
    { "  SVE",               "sve",    0xf5000000, 0xe5000000, COUNT_CLASS},
    { "  SVE",               "sve",    0xf5000000, 0xa5000000, COUNT_CLASS},
    { "  SVE",               "sve",    0xf5000000, 0x05000000, COUNT_CLASS},
    { "  SVE",               "sve",    0xf4000000, 0xa4000000, COUNT_CLASS},
    { "  SVE",               "sve",    0xf4000000, 0xe4000000, COUNT_CLASS}, // st1b
    { "  SVE",               "sve",    0xf4000000, 0x04000000, COUNT_CLASS},
    /* Loads and Stores */
    { "  AdvSimd ldstmult",  "advlsm", 0xbfbf0000, 0x0c000000, COUNT_CLASS}, // Advanced SIMD load/store multiple
    { "  AdvSimd ldstmult++", "advlsmp", 0xbfb00000, 0x0c800000, COUNT_CLASS}, // Advanced SIMD load/store multiple post-increment
    { "  AdvSimd ldst",      "advlss", 0xbf9f0000, 0x0d000000, COUNT_CLASS}, // Advanced SIMD load/store single
    { "  AdvSimd ldst++",    "advlssp", 0xbf800000, 0x0d800000, COUNT_CLASS}, // Advanced SIMD load/store single post-increment
    { "NEON arith",          "neonarith", 0xff000000, 0x4e000000, COUNT_CLASS}, // NEON arithmetic
    { "NEON arith",          "neonarith", 0xff000000, 0x6e000000, COUNT_CLASS}, // NEON arithmetic
    { "NEON arith",          "neonarith", 0xff000000, 0x2e000000, COUNT_CLASS}, // NEON arithmetic
    { "NEON arith",          "neonarith", 0xff000000, 0x5e000000, COUNT_CLASS}, // NEON arithmetic
    { "NEON arith",          "neonarith", 0xff000000, 0x0e000000, COUNT_CLASS}, // NEON arithmetic
    { "NEON logic",          "neonlogic", 0xff000000, 0x0f000000, COUNT_CLASS}, // NEON logic
    { "NEON logic",          "neonlogic", 0xff000000, 0x2f000000, COUNT_CLASS}, // NEON logic
    { "NEON logic",          "neonlogic", 0xff000000, 0x4f000000, COUNT_CLASS}, // NEON logic
    { "NEON logic",          "neonlogic", 0xff000000, 0x6f000000, COUNT_CLASS}, // NEON logic
    // Add more as needed
    /* Unclassified */
    { "Unclassified",        "unclas", 0x00000000, 0x00000000, COUNT_CLASS},
};


static InsnClassExecCount riscv_insn_classes[] = {
    // { "flw/fld",                            "addi",  0x0000007f, 0x00000007, COUNT_CLASS},
    // { "fsw/fsd",                            "addi",  0x0000007f, 0x00000027, COUNT_CLASS},
    // { "branch",                             "addi",  0x00000073, 0x00000063, COUNT_CLASS},
    // { "Fence",                              "addi",  0x0000007f, 0x0000000f, COUNT_CLASS},
    // { "Csr",                                "addi",  0x0000007f, 0x00000073, COUNT_CLASS},
    // { "lb",                                 "addi",  0x0000707f, 0x00000003, COUNT_CLASS},
    // { "lh",                                 "addi",  0x0000707f, 0x00001003, COUNT_CLASS},
    // { "lw",                                 "addi",  0x0000707f, 0x00002003, COUNT_CLASS},
    // { "ld",                                 "addi",  0x0000707f, 0x00003003, COUNT_CLASS},
    // { "lbu",                                "addi",  0x0000707f, 0x00004003, COUNT_CLASS},
    // { "lhu",                                "addi",  0x0000707f, 0x00005003, COUNT_CLASS},
    // { "lwu",                                "addi",  0x0000707f, 0x00006003, COUNT_CLASS},
    // { "sb",                                 "addi",  0x0000707f, 0x00000023, COUNT_CLASS},
    // { "sh",                                 "addi",  0x0000707f, 0x00001023, COUNT_CLASS},
    // { "sw",                                 "addi",  0x0000707f, 0x00002023, COUNT_CLASS},
    // { "sd",                                 "addi",  0x0000707f, 0x00003023, COUNT_CLASS},
    { "vector load",                         "addi",  0x0000007f, 0x00000007, COUNT_CLASS},
    { "vector store",                        "addi",  0x0000007f, 0x00000027, COUNT_CLASS},
    { "vector Arithmetic & Configuration",    "addi",  0x0000007f, 0x00000057, COUNT_CLASS},
    { "th.ldd",                             "addi",  0xf800707f, 0xf800400b, COUNT_CLASS},
    { "th.lwd",                             "addi",  0xf800707f, 0xe000400b, COUNT_CLASS},
    { "th.lwud",                            "addi",  0xf800707f, 0xf000400b, COUNT_CLASS},
    { "th.sdd",                             "addi",  0xf800707f, 0xf800500b, COUNT_CLASS},
    { "th.swd",                             "addi",  0xf800707f, 0xe000500b, COUNT_CLASS},
    { "sh1add",                             "addi",  0xfe00707f, 0x20002033, COUNT_CLASS},
    { "sh1add.uw",                          "addi",  0xfe00707f, 0x2000203b, COUNT_CLASS},
    { "sh2add",                             "addi",  0xfe00707f, 0x20004033, COUNT_CLASS},
    { "sh2add.uw",                          "addi",  0xfe00707f, 0x2000403b, COUNT_CLASS},
    { "sh3add",                             "addi",  0xfe00707f, 0x20006033, COUNT_CLASS},
    { "sh3add.uw",                          "addi",  0xfe00707f, 0x2000603b, COUNT_CLASS},
    /* Unclassified */
    { "Unclassified",                       "unclas", 0x00000000, 0x00000000, COUNT_CLASS},
};


static InsnClassExecCount sparc32_insn_classes[] = {
    { "Call",                "call",   0xc0000000, 0x40000000, COUNT_CLASS},
    { "Branch ICond",        "bcc",    0xc1c00000, 0x00800000, COUNT_CLASS},
    { "Branch Fcond",        "fbcc",   0xc1c00000, 0x01800000, COUNT_CLASS},
    { "SetHi",               "sethi",  0xc1c00000, 0x01000000, COUNT_CLASS},
    { "FPU ALU",             "fpu",    0xc1f00000, 0x81a00000, COUNT_CLASS},
    { "ALU",                 "alu",    0xc0000000, 0x80000000, COUNT_CLASS},
    { "Load/Store",          "ldst",   0xc0000000, 0xc0000000, COUNT_CLASS},
    /* Unclassified */
    { "Unclassified",        "unclas", 0x00000000, 0x00000000, COUNT_INDIVIDUAL},
};

static InsnClassExecCount sparc64_insn_classes[] = {
    { "SetHi & Branches",     "op0",   0xc0000000, 0x00000000, COUNT_CLASS},
    { "Call",                 "op1",   0xc0000000, 0x40000000, COUNT_CLASS},
    { "Arith/Logical/Move",   "op2",   0xc0000000, 0x80000000, COUNT_CLASS},
    { "Arith/Logical/Move",   "op3",   0xc0000000, 0xc0000000, COUNT_CLASS},
    /* Unclassified */
    { "Unclassified",        "unclas", 0x00000000, 0x00000000, COUNT_INDIVIDUAL},
};

/* Default matcher for currently unclassified architectures */
static InsnClassExecCount default_insn_classes[] = {
    { "Unclassified",        "unclas", 0x00000000, 0x00000000, COUNT_INDIVIDUAL},
};

typedef struct {
    const char *qemu_target;
    InsnClassExecCount *table;
    int table_sz;
} ClassSelector;

static ClassSelector class_tables[] = {
    { "aarch64", aarch64_insn_classes, ARRAY_SIZE(aarch64_insn_classes) },
    { "riscv64", riscv_insn_classes, ARRAY_SIZE(riscv_insn_classes) },
    { "sparc",   sparc32_insn_classes, ARRAY_SIZE(sparc32_insn_classes) },
    { "sparc64", sparc64_insn_classes, ARRAY_SIZE(sparc64_insn_classes) },
    { NULL, default_insn_classes, ARRAY_SIZE(default_insn_classes) },
};

static InsnClassExecCount *class_table;
static int class_table_sz;

static gint cmp_exec_count(gconstpointer a, gconstpointer b)
{
    InsnExecCount *ea = (InsnExecCount *) a;
    InsnExecCount *eb = (InsnExecCount *) b;
    return ea->count > eb->count ? -1 : 1;
}

static void free_record(gpointer data)
{
    InsnExecCount *rec = (InsnExecCount *) data;
    g_free(rec->insn);
    g_free(rec);
}

static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    g_autoptr(GString) report = g_string_new("Instruction Classes:\n");
    int i;
    GList *counts;
    InsnClassExecCount *class = NULL;

    for (i = 0; i < class_table_sz; i++) {
        class = &class_table[i];
        switch (class->what) {
        case COUNT_CLASS:
            if (class->count || verbose) {
                g_string_append_printf(report, "Class: %-24s\t(%ld hits)\n",
                                       class->class,
                                       class->count);
            }
            break;
        case COUNT_INDIVIDUAL:
            g_string_append_printf(report, "Class: %-24s\tcounted individually\n",
                                   class->class);
            break;
        case COUNT_NONE:
            g_string_append_printf(report, "Class: %-24s\tnot counted\n",
                                   class->class);
            break;
        default:
            break;
        }
    }

    counts = g_hash_table_get_values(insns);
    if (counts && g_list_next(counts)) {
        g_string_append_printf(report, "Individual Instructions:\n");
        counts = g_list_sort(counts, cmp_exec_count);

        for (i = 0; i < limit && g_list_next(counts);
             i++, counts = g_list_next(counts)) {
            InsnExecCount *rec = (InsnExecCount *) counts->data;
            g_string_append_printf(report,
                                   "Instr: %-24s\t(%ld hits)\t(op=0x%08x/%s)\n",
                                   rec->insn,
                                   rec->count,
                                   rec->opcode,
                                   rec->class ?
                                   rec->class->class : "un-categorised");
        }
        g_list_free(counts);
    }

    g_hash_table_destroy(insns);

    qemu_plugin_outs(report->str);
}

static void plugin_init(void)
{
    insns = g_hash_table_new_full(NULL, g_direct_equal, NULL, &free_record);
}

static void vcpu_insn_exec_before(unsigned int cpu_index, void *udata)
{
    uint64_t *count = (uint64_t *) udata;
    (*count)++;
}

static uint64_t *find_counter(struct qemu_plugin_insn *insn)
{
    int i;
    uint64_t *cnt = NULL;
    uint32_t opcode;
    InsnClassExecCount *class = NULL;

    /*
     * We only match the first 32 bits of the instruction which is
     * fine for most RISCs but a bit limiting for CISC architectures.
     * They would probably benefit from a more tailored plugin.
     * However we can fall back to individual instruction counting.
     */
    opcode = *((uint32_t *)qemu_plugin_insn_data(insn));

    for (i = 0; !cnt && i < class_table_sz; i++) {
        class = &class_table[i];
        uint32_t masked_bits = opcode & class->mask;
        if (masked_bits == class->pattern) {
            break;
        }
    }

    g_assert(class);

    switch (class->what) {
    case COUNT_NONE:
        return NULL;
    case COUNT_CLASS:
        return &class->count;
    case COUNT_INDIVIDUAL:
    {
        InsnExecCount *icount;

        g_mutex_lock(&lock);
        icount = (InsnExecCount *) g_hash_table_lookup(insns,
                                                       GUINT_TO_POINTER(opcode));

        if (!icount) {
            icount = g_new0(InsnExecCount, 1);
            icount->opcode = opcode;
            icount->insn = qemu_plugin_insn_disas(insn);
            icount->class = class;

            g_hash_table_insert(insns, GUINT_TO_POINTER(opcode),
                                (gpointer) icount);
        }
        g_mutex_unlock(&lock);

        return &icount->count;
    }
    default:
        g_assert_not_reached();
    }

    return NULL;
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    size_t i;

    for (i = 0; i < n; i++) {
        uint64_t *cnt;
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        cnt = find_counter(insn);

        if (cnt) {
            if (do_inline) {
                qemu_plugin_register_vcpu_insn_exec_inline(
                    insn, QEMU_PLUGIN_INLINE_ADD_U64, cnt, 1);
            } else {
                qemu_plugin_register_vcpu_insn_exec_cb(
                    insn, vcpu_insn_exec_before, QEMU_PLUGIN_CB_NO_REGS, cnt);
            }
        }
    }
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv)
{
    int i;

    /* Select a class table appropriate to the guest architecture */
    for (i = 0; i < ARRAY_SIZE(class_tables); i++) {
        ClassSelector *entry = &class_tables[i];
        if (!entry->qemu_target ||
            strcmp(entry->qemu_target, info->target_name) == 0) {
            class_table = entry->table;
            class_table_sz = entry->table_sz;
            break;
        }
    }

    for (i = 0; i < argc; i++) {
        char *p = argv[i];
        g_autofree char **tokens = g_strsplit(p, "=", -1);
        if (g_strcmp0(tokens[0], "inline") == 0) {
            if (!qemu_plugin_bool_parse(tokens[0], tokens[1], &do_inline)) {
                fprintf(stderr, "boolean argument parsing failed: %s\n", p);
                return -1;
            }
        } else if (g_strcmp0(tokens[0], "verbose") == 0) {
            if (!qemu_plugin_bool_parse(tokens[0], tokens[1], &verbose)) {
                fprintf(stderr, "boolean argument parsing failed: %s\n", p);
                return -1;
            }
        } else if (g_strcmp0(tokens[0], "count") == 0) {
            char *value = tokens[1];
            int j;
            CountType type = COUNT_INDIVIDUAL;
            if (*value == '!') {
                type = COUNT_NONE;
                value++;
            }
            for (j = 0; j < class_table_sz; j++) {
                if (strcmp(value, class_table[j].opt) == 0) {
                    class_table[j].what = type;
                    break;
                }
            }
        } else {
            fprintf(stderr, "option parsing failed: %s\n", p);
            return -1;
        }
    }

    plugin_init();

    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
