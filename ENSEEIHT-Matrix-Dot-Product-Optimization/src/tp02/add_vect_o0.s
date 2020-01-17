	.file	"add_vect.c"
	.section	.rodata
.LC1:
	.string	" %f "
	.align 8
.LC2:
	.string	" Temps d'execution %d secondes %d millisecondes \n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$832, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movl	%edi, -836(%rbp)
	movq	%rsi, -848(%rbp)
	movl	$0, %ebx
	jmp	.L2
.L5:
	movl	%ebx, %eax
	testq	%rax, %rax
	js	.L3
	pxor	%xmm3, %xmm3
	cvtsi2ssq	%rax, %xmm3
	movd	%xmm3, %eax
	jmp	.L4
.L3:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	movaps	%xmm0, %xmm2
	addss	%xmm0, %xmm2
	movd	%xmm2, %eax
.L4:
	movl	%ebx, %edx
	movl	%eax, -432(%rbp,%rdx,4)
	movl	%ebx, %edx
	movl	.LC0(%rip), %eax
	movl	%eax, -832(%rbp,%rdx,4)
	addl	$1, %ebx
.L2:
	cmpl	$99, %ebx
	jbe	.L5
	call	clock
	movq	%rax, -24(%rbp)
	movl	$0, %r12d
	jmp	.L6
.L9:
	movl	$0, %ebx
	jmp	.L7
.L8:
	movl	%ebx, %eax
	movss	-432(%rbp,%rax,4), %xmm1
	movl	%ebx, %eax
	movss	-832(%rbp,%rax,4), %xmm0
	addss	%xmm0, %xmm1
	movd	%xmm1, %eax
	movl	%ebx, %edx
	movl	%eax, -432(%rbp,%rdx,4)
	addl	$1, %ebx
.L7:
	cmpl	$99, %ebx
	jbe	.L8
	addl	$1, %r12d
.L6:
	cmpl	$19999999, %r12d
	jbe	.L9
	call	clock
	subq	-24(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	$0, %ebx
	jmp	.L10
.L11:
	movl	%ebx, %eax
	movss	-432(%rbp,%rax,4), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movl	$.LC1, %edi
	movl	$1, %eax
	call	printf
	addl	$10, %ebx
.L10:
	cmpl	$99, %ebx
	jbe	.L11
	movl	$10, %edi
	call	putchar
	movq	-32(%rbp), %rax
	imulq	$1000, %rax, %rcx
	movabsq	$4835703278458516699, %rdx
	movq	%rcx, %rax
	imulq	%rdx
	sarq	$18, %rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	movl	%eax, %ebx
	movl	$274877907, %edx
	movl	%ebx, %eax
	mull	%edx
	movl	%edx, %ecx
	shrl	$6, %ecx
	imull	$1000, %ecx, %eax
	movl	%ebx, %ecx
	subl	%eax, %ecx
	movl	$274877907, %edx
	movl	%ebx, %eax
	mull	%edx
	movl	%edx, %eax
	shrl	$6, %eax
	movl	%ecx, %edx
	movl	%eax, %esi
	movl	$.LC2, %edi
	movl	$0, %eax
	call	printf
	movl	$0, %eax
	addq	$832, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC0:
	.long	1010055512
	.ident	"GCC: (Debian 4.9.2-10+deb8u1) 4.9.2"
	.section	.note.GNU-stack,"",@progbits
