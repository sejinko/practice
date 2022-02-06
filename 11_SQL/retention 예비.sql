
-- ���⼭ date(payment_date) ������ �ذ��ؾ����� �� �������� �� �� ����






-- DAU - �ű�, ����, ��Ż ����ϱ�
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	-- �ű�����
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- ��������
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN,
	-- ���ͱ��� ��Ż ��
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day,
	-- ��Ż����
	case
		when  
			(ifnull(strftime('%j', lead(date(payment_date)) over(partition by customer_id order by date(payment_date))), max(payment_date) over())) > 14
			- strftime('%j',date(payment_date))
		then 1 else 0
	end out_YN
FROM 
	payment;
	
	
-- ���̺� ���� �� �ε��� ????????????????????? �̰� �� �𸣰���
alter table #dau_churn rebuild with (data_compression = page );
raiserror('���̺� ���� �Ϸ�' , 0 , 1) with nowait ;

create unique clustered index ucix on #dau_churn(account_id, std_dt)
with (data_compression = page);
raiserror('���̺� �ε��� �Ϸ�' , 0 , 1) with nowait ;


-- ������ ����ϱ� ??????????????????? �̰͵� �� �𸣰���
SELECT 
	max(payment_date),
	min(payment_date)
from
	payment;

SELECT 
	last_value(payment_date) over()
from
	payment
order by
	payment_date desc;



-- �ű�,����,�������� retention
with tm1 as
(
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	-- �ű�����
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- ��������
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN,
	-- ���ͱ��� ��Ż ��
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day,
	-- ��Ż����
	case
		when 
			(ifnull(strftime('%j', lead(date(payment_date)) over(partition by customer_id order by date(payment_date))), max(payment_date) over())) > 14
			- strftime('%j',date(payment_date))
		then 1 else 0
	end out_YN
FROM 
	payment
)
,
with x as
(
    select
	    date(payment_date),
	    customer_id,
	    new_YN,
	    comeback_YN
    from 
    	tm1
    where 1=1
    and date(payment_date) <= max(payment_date) over()
),
y as (
    select 
    	date(payment_date),
    	new_YN,
    	comeback_YN,
    	count(distinct customer_id)
    from tm1
    where 1=1
    and date(payment_date) <= max(payment_date) over()
    group by
    	date(payment_date),
    	new_YN,
    	comeback_YN
),
 z as
(
    select 
    	date(payment_date),
    	new_YN,
    	comeback_YN, 
    	(strftime('%j', post.date(payment_date)) - strftime('%j', date(payment_date))) as day_n, -- ������ �յڰ� �ٲ� �� ����
        count(distinct post.customer_id ) as visit , max(y.n) as total
    from x
    join x as post 
    	on x.customer_id = post.customer_id 
    	and post.date(payment_date) > x.date(payment_date)
    join y 
    	on x.date(payment_date) = y.date(payment_date) 
    	and x.new_YN = y.new_YN and x.comeback_YN = y.comeback_YN
    where 1=1
    and (strftime('%j',post.date(payment_date)) - strftime(x.date(payment_date))) <= 31
    group by
    	x.date(payment_date),
    	x.new_YN,
    	x.comeback_YN,
    	strftime('%j', post.date(payment_date)) - strftime('%j', date(payment_date))
)
select 
	date(payment_date),
	new_YN,
	comeback_YN,
	day_n,
	visit,
	total,
	(visit / (total *1.)) * 100 retention
into #retention
from z	

SELECT *
from #retention


-- ���űݾ� �� retention(�ű�,����,�������� retention���� ���űݾ׺� �÷��� ����� ���� �Ϳ� �Ұ� ��)
if object_id('tempdb..#retention_buy') is not null
	drop table #retention_buy	
	
with x as (
    select 
    	a.date(payment_date),
    	a.customer_id,
    	a.new_YN,
    	a.comeback_YN,
    	amount,
    	case 
    		when amount = 0 then '01. 0'
			when amount <= 30000 then '02. <= 30000'
			when amount <= 100000 then '03. <= 100000'
			when amount > 100000 then '04. 100000 >'
		else '01. 0' end as amount_group
    		from tm1 a
    where 1=1
    and a.date(payment_date) <= max(a.date(payment_date)
)
,
y as (
    select 
    	b.date(payment_date),
    	b.new_yn,
    	b.comeback_yn,
    	case
    		when b.amount = 0 then '01. 0'
			when b.amount <= 30000 then '02. <= 30000'
			when b.amount <= 100000 then '03. <= 100000'
			when b.amount > 100000 then '04. 100000 >'
			else '01. 0' 
		end as amount_group,
		count(distinct b.customer_id) as n
    from 
    	tm1 b
    where 1=1
    and b.date(payment_date) <= max(b.payment_date)
    group by 
    	b.date(payment_date),
    	new_YN,
    	comeback_YN,
    	case
    		when amount = 0 then '01. 0'
			when amount <= 30000 then '02. <= 30000'
			when amount <= 100000 then '03. <= 100000'
			when amount > 100000 then '04. 100000 >'
			else '01. 0' 
		end
),
 z as
(
    select 
    	x.date(payment_date),
    	x.new_YN,
    	x.comeback_YN, 
    	x.(strftime('%j', post.date(payment_date)) - strftime('%j', date(payment_date))) as day_n, -- ������ �յڰ� �ٲ� �� ����
        count(distinct post.customer_id ) as visit , max(y.n) as total
    from x
    join x as post 
    	on x.customer_id = post.customer_id 
    	and post.date(payment_date) > x.date(payment_date)
    join y 
    	on x.date(payment_date) = y.date(payment_date) 
    	and x.new_YN = y.new_YN and x.comeback_YN = y.comeback_YN
    	and x.amount_group = y.amount_group -- �ű�,����,�������� retention �� �ٸ��� �߰��� �ڵ�
    where 1=1
    and (strftime('%j',post.date(payment_date)) - strftime(x.date(payment_date))) <= 31
    group by
    	x.date(payment_date),
    	x.new_YN,
    	x.comeback_YN,
    	amount_group, -- �ű�,����,�������� retention�� �ٸ��� �߰��� �ڵ�
    	strftime('%j', post.date(payment_date)) - strftime('%j', date(payment_date))
)
select
	date(payment_date),
	new_YN,
	comeback_YN,
	amount_group,
	day_n,
	visit,
	total,
	(visit / (total *1.)) * 100 retention
into #retention_buy
from z


SELECT *
FROM #retention_buy
